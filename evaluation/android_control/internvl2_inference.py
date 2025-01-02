import argparse
import itertools
import json
import os
import random
import time
from functools import partial
import torch
from internvl.model.internvl_chat import InternVLChatModel
from internvl.train.dataset import build_transform, dynamic_preprocess
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ds_collections = {
    'ac_high': {
        'root': '/path/to/images',
        'annotation': 'eval_json_files/ac_high_processing.jsonl',
        'max_new_tokens': 999,
        'min_new_tokens': 1,
    },
    'ac_low': {
        'root': '/path/to/images',
        'annotation': 'eval_json_files/ac_low_processing.jsonl',
        'max_new_tokens': 999,
        'min_new_tokens': 1,
    }

}


def collate_fn(batches, tokenizer):
    pixel_values = torch.cat([_['pixel_values'] for _ in batches], dim=0)
    questions = [_['question'] for _ in batches]
    items = [_['item'] for _ in batches]
    return pixel_values, questions, items


class AndroidControlDataset(torch.utils.data.Dataset):

    def __init__(self, root, annotation, input_size=224, dynamic_image_size=False,
                 use_thumbnail=False, max_num=6):
        self.root = root
        self.items = []
        f = open(annotation)
        data = f.readlines()
        for data_line in data:
            data_line = json.loads(data_line)
            self.items.append(data_line)
        self.input_size = input_size  # input size??
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.transform = build_transform(is_train=False, input_size=input_size)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image_path, question = item['image'], item['conversations'][0]['value']
        image = Image.open(image_path).convert('RGB')
        if self.dynamic_image_size:  
            images = dynamic_preprocess(image, image_size=self.input_size,
                                        use_thumbnail=self.use_thumbnail,
                                        max_num=self.max_num)
        else:
            images = [image]
        pixel_values = [self.transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        return {
            'question': question,
            'pixel_values': pixel_values,
            'item': item,
        }



class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)



def evaluate_chat_model():
    random.seed(args.seed)

    if args.ds_name_list == None:
        ds_names = ds_collections.keys()
    else:
        ds_names = args.ds_name_list
    for ds_name in ds_names:
        dataset = AndroidControlDataset(
            root=ds_collections[ds_name]['root'],
            annotation=ds_collections[ds_name]['annotation'],
            input_size=image_size,
            dynamic_image_size=args.dynamic,
            use_thumbnail=use_thumbnail,
            max_num=args.max_num
        )
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=InferenceSampler(len(dataset)),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=partial(collate_fn, tokenizer=tokenizer),
        )


        logger.info(f'Evaluating {ds_name} ...')

        outputs = []
        for _, (pixel_values, questions, items) in tqdm(enumerate(dataloader)):
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
            generation_config = dict(
                num_beams=args.num_beams,
                max_new_tokens=ds_collections[ds_name]['max_new_tokens'],
                min_new_tokens=ds_collections[ds_name]['min_new_tokens'],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
            )
            pred = model.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=questions[0],
                generation_config=generation_config
            )
            preds = [pred]

            for question, answer, item in zip(questions, preds, items):
                question_id = item['id']
                image_id = item['image'].split("\/")[-1].replace(".png", "")
                text = question
                output = {
                    'question_id': question_id,
                    'image_id': image_id,
                    'prompt': text,
                    'pred': answer,
                    'model_id': model_id,
                    'metadata': {},
                    "task_name": "task2action",
                    "string_format": "CSV_String",
                    "position_format": "related"
                }
                outputs.append(output)

        torch.distributed.barrier()
        world_size = torch.distributed.get_world_size()
        merged_outputs = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

        merged_outputs = [json.loads(_) for _ in merged_outputs]
        merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

        torch.distributed.barrier()

        if torch.distributed.get_rank() == 0:
            print(len(merged_outputs))
            results_file = f'{model_id}_{ds_name}.jsonl'
            results_file = os.path.join(args.out_dir, results_file)
            with open(results_file, 'w') as f:
                for output in merged_outputs:
                    json.dump(output, f)
                    f.write('\n')
            print('Results saved to {}'.format(results_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--datasets', type=str, default='')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--num-beams', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    parser.add_argument('--ds_name_list', type=str, nargs='*', default=None, help='List of dataset names')
    args = parser.parse_args()

    args.datasets = args.datasets.split(',')
    logger.info('datasets:', args.datasets)
    assert args.batch_size == 1, 'Only batch size 1 is supported'

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))


    if args.auto:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    kwargs = {'device_map': 'auto'} if args.auto else {}
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True, use_fast=False)
    model = InternVLChatModel.from_pretrained(
        args.checkpoint, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16,
        load_in_8bit=args.load_in_8bit, **kwargs).eval()
    if not args.load_in_8bit and not args.auto:
        model = model.cuda()
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = model.config.use_thumbnail

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    if total_params > 20 or args.dynamic:
        args.num_beams = 1
        logger.info(f'[test] total_params: {total_params}B, use num_beams: {args.num_beams}')
    else:
        logger.info(f'[test] total_params: {total_params}B')
    logger.info(f'[test] image_size: {image_size}')
    logger.info(f'[test] template: {model.config.template}')
    logger.info(f'[test] dynamic_image_size: {args.dynamic}')
    logger.info(f'[test] use_thumbnail: {use_thumbnail}')
    logger.info(f'[test] max_num: {args.max_num}')
    if torch.distributed.get_rank() == 0:
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)

    model_id = '_'.join(args.checkpoint.split('/')[-1:])
    evaluate_chat_model()