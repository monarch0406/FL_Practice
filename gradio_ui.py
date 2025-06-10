"""
gradio_ui.py ─ Federated-Learning GUI + TensorBoard
---------------------------------------------------
Docker 範例：
    docker build -t fflab-app .
    docker run -it --rm -p 7860:7860 -p 6006:6006 fflab-app
"""

import argparse, time, threading, subprocess
import gradio as gr


# ──────────────────────────────────────────────────────────
# 1. 背景啟動 TensorBoard
# ──────────────────────────────────────────────────────────
def launch_tensorboard(log_dir: str, port: int = 6006) -> None:
    def _run():
        subprocess.run(
            ["tensorboard", "--logdir", log_dir,
             "--port", str(port), "--host", "0.0.0.0"],
            check=False,
        )
    threading.Thread(target=_run, daemon=True).start()
    time.sleep(2)  # 等 TensorBoard 起來


# ──────────────────────────────────────────────────────────
# 2. 建立 Gradio UI
# ──────────────────────────────────────────────────────────
def launch_gradio_ui(main_function, args: argparse.Namespace):

    with gr.Blocks() as demo:
        gr.Markdown("# Federated Learning GUI")

        # --------- FL Setting ----------
        gr.Markdown("## FL Setting")
        with gr.Row():
            fl_inputs1 = {
                "algorithm": gr.Dropdown(
                    ["fedavg", "fedprox", "scaffold", "fedmd", "feddf",
                     "fedgen", "moon", "fedproto", "fpl", "flwf",
                     "cfed", "fedcl"],
                    value="fedavg", label="Algorithm"
                ),
                "dataset": gr.Dropdown(
                    ["Mnist", "Cifar10", "Digits", "Office-Caltech"],
                    value="Mnist", label="Dataset"
                ),
                "skew_type": gr.Dropdown(
                    ["label", "feature", "quantity"],
                    value="label", label="Skew Type"
                ),
                "alpha": gr.Number(value=100.0, label="Alpha"),
                "model": gr.Dropdown(
                    ["SimpleCNN", "MyCNN", "ResNet10"],
                    value="SimpleCNN", label="Model"
                ),
                "batch_size": gr.Slider(32, 512, value=128,
                                        step=32, label="Batch Size"),
            }

        with gr.Row():
            fl_inputs2 = {
                "num_clients": gr.Slider(1, 20, value=10, step=1,
                                         label="Number of Clients"),
                "num_classes": gr.Slider(2, 20, value=10, step=1,
                                         label="Number of Classes"),
                "num_rounds": gr.Slider(1, 20, value=10, step=1,
                                        label="Number of Rounds"),
                "num_epochs": gr.Slider(1, 10, value=5, step=1,
                                        label="Number of Epochs"),
            }

        # --------- Dynamic Setting ----------
        gr.Markdown("## Dynamic Setting")
        with gr.Row():
            dp_inputs = {
                "dynamic_type": gr.Dropdown(
                    ["static", "round-robin", "incremental-arrival",
                     "incremental-departure", "random", "markov"],
                    value="static", label="Dynamic Type"),
                "round_start": gr.Number(value=1, label="Round Start"),
                "initial_clients": gr.Slider(1, 20, value=5, step=1,
                                             label="Initial Clients"),
                "interval": gr.Slider(5, 30, value=10, step=1,
                                      label="Interval"),
                "overlap_clients": gr.Slider(1, 10, value=2, step=1,
                                             label="Overlap Clients"),
                "dpfl": gr.Checkbox(False, label="Enable Knowledge Pool Module"),
            }

        # --------- 把所有元件收集起來 ----------
        args_name  = list(fl_inputs1.keys()) + list(fl_inputs2.keys()) + list(dp_inputs.keys())
        args_value = list(fl_inputs1.values()) + list(fl_inputs2.values()) + list(dp_inputs.values())

        # --------- 按鈕 / 輸出 ----------
        start_button = gr.Button("▶ Start Training")
        output_box   = gr.Textbox(label="Training Progress", lines=4)

        # --------- TensorBoard ----------
        with gr.Row():
            launch_tensorboard(args.log_dir)
            gr.HTML(
                "<iframe src='http://127.0.0.1:6006' width='100%' "
                "height='700' style='border:none;'></iframe>"
            )

        # --------- State & Callback ----------
        is_running = gr.State(False)

        def _on_click(running, *vals):
            if running:
                return "⚠️ 訓練已在進行中，請稍候…", running

            running = True
            for k, v in dict(zip(args_name, vals)).items():
                setattr(args, k, v)

            main_function(args)
            running = False
            return "✅ Training complete! 請至 TensorBoard 查看結果。", running

        start_button.click(
            _on_click,
            inputs=[is_running] + args_value,
            outputs=[output_box, is_running]
        )

        print("➡  打開瀏覽器：http://localhost:7860", flush=True)
        # --------- 啟用 queue，限制同時只跑一個 Job ----------
        demo.queue()

    # --------- 啟動 Server ----------
    
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


# ──────────────────────────────────────────────────────────
# 3. 測試用 dummy main()
# ──────────────────────────────────────────────────────────
def dummy_train(cfg):
    import random, time
    for r in range(cfg.num_rounds):
        time.sleep(0.3)
        print(f"[Round {r+1}/{cfg.num_rounds}] acc={random.random():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_rounds", type=int, default=5)
    parser.add_argument("--log_dir", type=str, default="runs")
    default_args = parser.parse_args()

    launch_gradio_ui(dummy_train, default_args)
    

