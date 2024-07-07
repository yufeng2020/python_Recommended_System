# 定义一个函数来进行加法运算
def add(a, b):
    return a + b

# 定义一个函数来进行减法运算
def subtract(a, b):
    return a - b

# 定义一个函数来进行字符串反转
def reverse_string(s):
    return s[::-1]

# 定义一个函数来查找列表中的最大值
def find_max(lst):
    return max(lst)

# 定义一个函数来读取文件内容
def read_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        return "文件未找到"

# 定义一个函数来写入内容到文件
def write_file(file_path, content):
    with open(file_path, 'w') as file:
        file.write(content)
    return "写入成功"

# 主函数来调用其他函数
def main():
    # 调用加法函数
    result_add = add(10, 5)
    print(f"10 + 5 = {result_add}")

    # 调用减法函数
    result_subtract = subtract(10, 5)
    print(f"10 - 5 = {result_subtract}")

    # 调用字符串反转函数
    original_string = "hello"
    reversed_string = reverse_string(original_string)
    print(f"{original_string} 反转后是 {reversed_string}")

    # 调用查找最大值函数
    numbers = [1, 2, 3, 4, 5]
    max_number = find_max(numbers)
    print(f"列表 {numbers} 中的最大值是 {max_number}")

    # 调用文件读取函数
    file_path = 'example.txt'
    file_content = read_file(file_path)
    print(f"文件内容:\n{file_content}")

    # 调用文件写入函数
    write_message = "这是一个测试内容。"
    write_result = write_file(file_path, write_message)
    print(write_result)

# 调用主函数
if __name__ == "__main__":
    main()
