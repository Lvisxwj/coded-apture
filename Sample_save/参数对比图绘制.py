import matplotlib.pyplot as plt
import numpy as np

# 提供的数据
result_avg = np.array([-8.0223412e+01,  5.8504051e+04,  1.6125424e-01,  3.5990024e-01,
                        9.4366550e-02,  3.1615031e+00,  1.1392489e-04,  5.1483110e-04,
                        2.0404726e+02,  4.6738297e-01,  1.4411980e+00,  1.5834285e-01,
                        3.0493012e-01,  1.7502816e+03,  2.0828605e-01, -6.7756696e+00,
                        8.0676979e-01,  3.8410252e-01,  5.3720540e-01,  1.8222977e+03,
                        2.3476210e-01,  3.2102534e-01,  4.9723744e-01,  1.9992508e+01,
                        2.1723662e+03,  7.4753296e+02,  2.1323769e+00, -9.5020793e-02,
                        -6.5387213e-01,  3.8854097e-04,  6.2792039e-01,  4.5428428e-01])

label_avg = np.array([-4.6742404e+02,  1.5745716e+05,  9.2956796e-02,  5.5549610e-01,
                       1.8619436e-01,  5.9808989e+00,  3.5324975e-04,  9.2462241e-04,
                       4.4578214e+02,  5.8475143e-01,  3.1875014e+00,  2.5227565e-01,
                       3.8986984e-01,  2.1554824e+03,  4.0968612e-01,  6.3262564e-01,
                       7.4497461e-01,  5.9919691e-01,  9.2801988e-01,  3.2934592e+03,
                       4.2413270e-01,  5.2512246e-01,  7.0589632e-01,  3.5041924e+01,
                       3.7795930e+03,  1.5316200e+03,  2.4302273e+00, -1.8732145e-01,
                       -6.1347812e-01,  5.7137810e-04,  5.7826352e-01,  3.5932574e-01])

result_avg = np.abs(result_avg)
label_avg = np.abs(label_avg)

# 通道数量
C = len(result_avg)
channels = np.arange(1, C+1)

# 绘制图像
plt.figure(figsize=(10, 6))  # 增大图像尺寸
plt.plot(channels, result_avg, color='blue', label='Result', linewidth=2)
plt.plot(channels, label_avg, color='red', label='Label', linewidth=2)

plt.xlabel('Channel', fontsize=14)
plt.ylabel('Average Value', fontsize=14)
plt.title('Comparison of Result and Label Averages per Channel', fontsize=16)

plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# 设置y轴为对数刻度
plt.yscale('log')

# 设置Y轴范围（根据实际数据情况可调整）
y_min = min(result_avg[result_avg > 0].min(), label_avg[label_avg > 0].min())  # 确保最小值大于0以适应对数刻度
y_max = max(result_avg.max(), label_avg.max())
plt.ylim(y_min, y_max)

plt.tight_layout()

# 保存图像
save_figure_path = "ours_compare_plot.png"
plt.savefig(save_figure_path, dpi=300)

# 显示图像
plt.show()