# 用 ComfyUI + Rapid-MLX，把一只纸船拍成 AI 微电影

一只红纸船载着一盏灯，从雨夜车站出发，经过云海、月球古城和发光森林，最后停在黎明的码头。这是《最后一盏灯》的故事：六个短镜头，固定的主角和配色，最后用片头片尾串成一部约 30 秒的短片。

这个 demo 把两种生成任务分开：Qwen-Image 2.1 画关键帧，LTX-2.5 从关键帧生成运动和声音。ComfyUI 负责节点连接、队列和结果预览，Rapid-MLX 提供本地模型服务，通过 MLX 执行 Apple Silicon 推理。

> 状态：六张关键帧已实际生成并人工检查；视频和成片等待下一段 GPU 窗口，尚未实跑。本文是待补充成片的草稿。

![六个已生成并审核的关键帧](media/contact-sheet.jpg)

六张图使用 1024×704、40 步，ComfyUI 执行耗时及 SHA-256 记录在 [verified-keyframes.json](verified-keyframes.json)。共享机器上的这些执行记录不作为受控性能基准。

## 先选一个适合多镜头生成的故事

独立生成的六个镜头很难自动保持复杂人物的脸、衣服和动作连续性。这次选一只红色折纸船作为主角：轮廓简单、颜色鲜明，一盏暖色灯笼贯穿全部画面。场景可以变化很大，观众仍能找到视觉锚点。

这是一种美术设计策略，并不能保证模型每次生成完全相同的纸船。分镜之间允许一定造型变化，重点是让颜色、材质、灯光和故事方向保持连续。实际选片仍需要人工判断。

`storyboard.json` 为每个镜头保存三类信息：关键帧提示词、运动与声音提示词、固定 seed。关键帧写场景和构图，视频提示词写镜头如何运动、物体如何运动，以及希望出现的环境声。例如雨夜车站的一镜：

```text
Low camera slowly tracks alongside the red paper boat as it drifts
across the rain puddle. Raindrops ripple the water, the amber lantern
flame gently flickers, mist moves through the deserted station.
Quiet rain and distant train rumble. One continuous shot.
```

每个镜头只安排一个主要运动，避免四秒内同时要求复杂转场、角色动作和场景变换。标题和品牌文字在后期确定性绘制，让模型专注画面。

## ComfyUI 的 IMAGE 怎样变成 VIDEO

```text
第一遍：RapidMLXImage → SaveImage → 审核关键帧
第二遍：LoadImage → RapidMLXVideo → SaveVideo → 审核镜头
最后：FFmpeg 剪辑、保留镜头原声、添加片头片尾
```

生图节点沿用前一个[旅行海报 demo](../comfyui-travel/README.md)。它调用 Rapid-MLX 的 `/v1/images/generations`，将返回的 PNG 转换为 ComfyUI 的标准 IMAGE 张量。

新增的视频节点接收一张 IMAGE，将像素编码成 PNG，以 multipart 请求上传到 `/v1/videos`。这一步是实际的图生视频推理：相机运动、涟漪、雾和灯火变化由视频模型生成。后期剪辑负责组织镜头。

```bash
curl http://127.0.0.1:18428/v1/videos \
  -F model=ltx-2.5-mlx-q8 \
  -F 'prompt=Camera tracks alongside a red paper boat in the rain' \
  -F input_reference=@keyframe.png \
  -F size=704x480 -F frames=97 -F fps=24 -F seed=7300
```

创建请求返回 job ID。节点轮询 `/v1/videos/{id}`，等状态变为 `completed` 后，通过 `/v1/videos/{id}/content` 下载 MP4。完整节点见 [rapid_mlx_video](custom_nodes/rapid_mlx_video/__init__.py)。

视频节点返回 ComfyUI 原生的 `VIDEO` 对象，底层使用 `InputImpl.VideoFromFile`。这样 Save Video 可以接收编码视频及其音轨，避免把 MP4 拆成图片后丢失 LTX 生成的声音。对应契约见 [ComfyUI 视频节点源码](https://github.com/Comfy-Org/ComfyUI/blob/0df64eb242b7c5759c3e86afd5d1846d923b1033/comfy_extras/nodes_video.py)。

## 两个独立环境，各自负责自己的计算

ComfyUI 用 `--cpu` 启动，Torch 在这里传递像素和管理工作流。模型加载与 Metal 计算在独立的 Rapid-MLX 进程里完成。CPU 模式不意味着整条生成流程只使用 CPU。

先安装图片节点、视频节点，再启动模型服务。LTX-2.5 使用经过固定版本审核的独立 MLX runtime，具体安装和 commit 见 [视频部署指南](../../docs/guides/video-generation.md#ltx-25)，不要直接替换成任意最新版。

```bash
rapid-mlx serve qwen-image-2.1 --host 127.0.0.1 --port 18427

RAPID_MLX_LTX25_RUNTIME=/absolute/path/ltx-2-mlx/.venv/bin/ltx-2-mlx \
  rapid-mlx serve ltx-2.5-mlx-q8 --host 127.0.0.1 --port 18428 \
  --video-output-dir /absolute/path/cinema/media/server-jobs
```

这两个命令属于不同阶段。我们的运行方式是先完成关键帧，再正常关闭自己的图像服务，之后进入视频阶段，让模型驻留和生成顺序容易控制。共享机器上也要先协调 GPU 窗口。

LTX 模型包和独立 runtime 有各自的体积与内存要求。本文展示机是 256 GB 统一内存的 M3 Ultra；这套结果不能用来承诺小内存 Mac 的运行能力。转换模型的结构、量化和许可证见 [模型卡](https://huggingface.co/MrMofer/ltx-2.5-mlx-q8)。

## 批量制作，保留断点和证据

```bash
python3 examples/comfyui-cinema/batch.py keyframes \
  --output examples/comfyui-cinema/media --limit 6
python3 examples/comfyui-cinema/batch.py shots \
  --output examples/comfyui-cinema/media --limit 6
```

脚本每次只提交一个镜头，保存 Comfy prompt ID，再轮询执行历史。视频节点另存 Rapid job ID；相同图片、相同参数的重跑会继续查询原任务，而不是再次提交生成。最终文件用 SHA-256 校验。

这仍有明确的恢复边界：如果创建请求已经到达服务端，却在收到 job ID 前断线，客户端不能保证安全重提，必须先检查服务端任务。ComfyUI 重启丢失历史时，也要保留 receipt 并人工核对。服务端的 `--video-output-dir` 用于持久保存完成的视频任务。

本次每镜头使用 704×480、97 帧、24 fps，约 4.04 秒。LTX-2.5 的 distilled 路径要求合法的帧数和尺寸，也不接受 CFG guidance 或 negative prompt。修改参数前应读取 `/v1/videos/capabilities`。

## 后期只做可解释的剪辑

```bash
python3 examples/comfyui-cinema/render.py \
  --input examples/comfyui-cinema/media \
  --output examples/comfyui-cinema/media/the-last-lantern.mp4
```

脚本保留镜头原声，统一编码，加入各三秒的片头与片尾。输出为 1080p 的编辑放大版，生成镜头的原始分辨率仍是 704×480；这是两项不同的信息。片中标注 AI 生成。

最终发布前，需要检查纸船变形、背景突变、镜头运动是否成立，以及音轨是否适合成片。完整运行证据会包含模型和 runtime 版本、每个任务的执行记录、耗时与输出校验值。

这个流程的用途不限于纸船故事。换一份分镜和提示词，就可以制作产品概念片、虚构旅行预告或系列短内容；Rapid-MLX 承担本地模型服务，ComfyUI 将这些能力连接成可编辑、可重复运行的创作流程。
