# Task4 Auto VAE Pack

## 用法
把 OASIS `.nii` / `.nii.gz` 文件放到一个文件夹 (默认 `Pictures/`)。

### 运行
```bash
python task4_main.py --data_dir Pictures --epochs 5 --outdir results_task4
```

程序会自动：
1. 如果没有 dataset.npy → 从 `--data_dir` 预处理生成
2. 用 VAE 训练
3. 输出结果图片：
   - `loss_curve.png`
   - `vae_samples.png`
   - `recon_vs_input.png`
   - `vae_mri.pt`
