# GainPPO

严格六维状态、主/腕各11类Gain选择的普通SB3 PPO，两阶段Actor-only BC → 多λ独立Actor-Critic训练。源码、配置、文档均独立放在本目录。

输入固定顺序：`main_pearson, wrist_pearson, gripper_speed, eef_translation_speed, eef_rotation_speed, step_progress`，Box(6,) float32。输出默认MultiDiscrete([11,11])，Actor logits=22，Critic=1；两条独立64→64 Tanh MLP。`shared_gain=true`提供单11类兼容。无动作掩码，λ只用于奖励。

```bash
cd /data/qinshiyu/wyb/EmbodiedComp_workspace/PPO
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/gainppo-mpl
.venv/bin/python -m pytest -q
.venv/bin/python -m gainppo.smoke --output outputs/smoke
```

`smoke`生成明确标注的合成教师数据，做3轮小型BC、两个λ各32步PPO及匹配种子评估，输出独立checkpoint和示例图。目录已存在会报错，换新路径运行。**合成BPP/成功率不是真实Codec/EComp结果。**

本机测试环境`.venv`仅在PPO内安装新增包，以`parent_env.pth`只读复用`../envs/openpi`的torch/numpy等。换机器应自行创建独立环境：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
```

复用某个已有venv时，嵌套`venv --system-site-packages`不会自动继承它的包；此处的pth是本机专用，不应提交或复制。环境版本见`docs/test_results.md`。

真实流程先标定，再采集教师、BC、单λ或sweep；配置和命令见[实验协议](docs/experiment_protocol.md)。真实入口需要明确`--authorize-real`。本轮没有运行真实环境、加载大型权重、访问GPU或启动VLA服务。

- [审计](docs/code_audit.md)：实际接口、版本差异、尚未核实部分。
- [实现说明](docs/implementation_notes.md)：网络、参考方法、恢复边界。
- [测试报告](docs/test_results.md)：实际运行命令、通过项与未执行项。

`outputs/`、`.venv/`、权重、packet和数据默认忽略；输出均使用新目录，禁止覆盖旧实验。完整包BPP包含ECD1头，单路BPP包含单路ECG1流但不分摊共享ECD1头，因此单路加权均值与总BPP之间有包头差异。
