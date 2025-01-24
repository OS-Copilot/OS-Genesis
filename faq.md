# FAQ

Thank you for your interest in OS-Genesis. Below are some questions we have collected from emails, Hugging Face, and WeChat communications. We hope these can be helpful to you.

## When will the checkpoints and data be available?

We have already uploaded the checkpoints and evaluation code for AndroidControl. ~~The remaining checkpoints will be uploaded in the coming days~~ (done). Due to server bandwidth limitations, this may take some time. The data will also be available shortly.


## How About Desktop?

Q: Why haven’t you worked on PC/Desktop data? Is there a particular reason?

A:
We originally intended to cover PC, mobile, and web. In fact, our high-level reverse-synthesis process can also run on PC (we used [OSWorld](https://os-world.github.io/) as the dynamic environment). However, we decided not to continue on the PC side for the following reasons:
1.	Data collection on PC is too difficult for a model-based approach.
For instance, in [OSWorld](https://os-world.github.io/), the success rate for GPT-4o across most scenarios is <10%, which means the proportion of high-quality trajectories would be low. Ensuring quality would require a massive amount of data and a more rigorous TRM, making costly.

2.	Even after collecting trajectories, there are significant challenges in training:
    1. Length of a11ytree:
We use a11ytree, and on desktop the a11ytree is much longer than the mobile or web DOM. In training that involves multimodal information, it exceeds the context window of models like InternVL and Qwen.
    2. Instruction-following issues:
Currently, open-source VLMs face major problems with instruction-following on PC environments.