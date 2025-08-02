import os
import numpy as np

# 文件夹路径
folderPath = "/data4/zhuo-file/extracted_data/5nm_input/"

# 获取文件夹中的所有 .mat 文件
files = [f for f in os.listdir(folderPath) if f.endswith('.mat')]

# 按文件名的序号对文件进行排序
fileNumbers = [int(os.path.splitext(f)[0]) for f in files]
sorted_indices = np.argsort(fileNumbers)
files = [files[i] for i in sorted_indices]

# 限定处理的文件数量（前 200 个）
numFilesToProcess = min(200, len(files))

# 定义需要跳过的序号
skipFiles = [15, 43, 109]

# 初始化变量以保存每层的全局最大值和最小值
globalMaxValues = np.full(84, -np.inf)
globalMinValues = np.full(84, np.inf)

# 打开文件用于写入结果
try:
    with open("/data4/zhuo-file/fyc_file/Arti11/ATTU_Norm/HSI_summary_results.txt", 'w') as resultsFile:
        # 遍历文件
        for k in range(numFilesToProcess):
            # 获取当前文件名的序号
            fileNum = int(os.path.splitext(files[k])[0])

            # 打印当前处理的文件名
            print(f'正在处理文件: {files[k]}')

            # 检查是否跳过当前文件
            if fileNum in skipFiles:
                resultsFile.write(f'文件: {files[k]} 被跳过\n')
                continue

            # 构建文件路径
            filePath = os.path.join(folderPath, files[k])

            # 加载 .mat 文件
            from scipy.io import loadmat
            data = loadmat(filePath)

            # 假设数据存储在变量 'extracted_data' 中
            data = data['extracted_data']

            # 获取数据的尺寸
            W, H, numLayers = data.shape

            # 检查是否有某一层的最大值超过 16000
            skip_this_data = False
            for i in range(numLayers):
                layerData = data[:, :, i]
                layer_max = np.max(layerData)
                if layer_max > 16000:
                    skip_this_data = True
                    break

            if skip_this_data:
                resultsFile.write(f'文件: {files[k]} 因某层最大值超过 16000 被跳过\n')
                continue

            # 初始化当前文件每层的最大值和最小值数组
            maxValues = np.zeros(numLayers)
            minValues = np.zeros(numLayers)

            # 计算每层的最大值和最小值
            for i in range(numLayers):
                layerData = data[:, :, i]
                maxValues[i] = np.max(layerData)
                minValues[i] = np.min(layerData)

                # 更新全局最大值和最小值
                globalMaxValues[i] = max(globalMaxValues[i], maxValues[i])
                globalMinValues[i] = min(globalMinValues[i], minValues[i])

            # 写入当前文件的结果到文件
            resultsFile.write(f'文件: {files[k]}\n')
            for i in range(numLayers):
                resultsFile.write(f'第{i + 1}层，最大值：{maxValues[i]}，最小值：{minValues[i]}\n')
            resultsFile.write('\n')

        # 写入总结结果到文件
        resultsFile.write('总结结果:\n')
        for i in range(numLayers):
            resultsFile.write(f'第{i + 1}层，在所有文件中的最大值：{globalMaxValues[i]}，最小值：{globalMinValues[i]}\n')

    print('结果已保存到 HSI_summary_results.txt')
except Exception as e:
    print(f"出现错误: {e}")