import random




def get_list(lists, keynum, full_seq_len):
    # 最终结果的列表
    final_result = []

    for lst in lists:
        for one in lst:
            if int(one) < 12 or int(one) > (full_seq_len-12):
                lst.remove(one)

        if len(lst) >= keynum:
            # 如果列表长度大于等于8，则均匀选择8个元素
            # lst = sorted(random.sample(lst, 8))
            interval = len(lst) // keynum
            ind = list(range(keynum))
            ind = [x * interval for x in ind]
            lst  = [lst[i] for i in ind]
        else:
            # 如果列表长度小于8，则从0到500之间均匀抽样，使得总共有8个元素
            additional_elements_count = keynum - len(lst)
            test_lst = lst.copy()
            test_lst.insert(0,8)
            test_lst.append(full_seq_len-8)
            for _ in range(additional_elements_count):
                # 寻找两个元素之间的差异最大的位置
                max_diff_index = 0
                max_diff = 0
                for i in range(len(test_lst) - 1):
                    diff = test_lst[i + 1] - test_lst[i]
                    if diff<10:
                        continue
                    if diff > max_diff:
                        max_diff = diff
                        max_diff_index = i
                # print(max_diff_index)
                # print(max_diff)

                # 计算两个元素之间的中间值
                mid_value = (test_lst[max_diff_index] + test_lst[max_diff_index + 1]) // 2

                # if mid_value < 12:
                #     mid_value = 12
                # if mid_value > (full_seq_len-12):
                #     mid_value = (full_seq_len-12)
                # print(mid_value)
                lst.insert(max_diff_index, mid_value)
                test_lst.insert(max_diff_index+1, mid_value)
        final_result.append(lst)

    return final_result


if __name__ == 'main':
    # 原始列表
    lists = [[1, 100, 110, 125],
            [0, 12, 130, 410],
            [16, 17, 18, 19, 20, 21],
            [0, 12, 71, 130, 200, 270, 340, 410,500]
            ]
    final_result = get_list(lists)
    print(final_result)


# import numpy as np

# # 给定的列表
# original_list = [1, 100, 110, 126]

# max_diff_index = 0
# max_diff = 0
# for i in range(len(original_list) - 1):
#     diff = original_list[i + 1] - original_list[i]
#     if diff > max_diff:
#         max_diff = diff
#         max_diff_index = i

# # 要生成的8个均匀插值点（包括原始列表的端点）
# interpolation_points = np.linspace(original_list[0], original_list[-1], 4)
# interpolation_points = np.array(interpolation_points, dtype=np.int16)
# print(interpolation_points)

# # 使用numpy.interp进行线性插值
# interpolated_values = np.interp(interpolation_points, original_list, interpolation_points)

# print(interpolated_values)