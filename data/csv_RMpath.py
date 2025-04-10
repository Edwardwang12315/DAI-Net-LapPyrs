# --coding:utf-8--
# 原始数据文件路径
input_file = "wider_face_train.txt"
# 修改后的数据保存路径
output_file = "wider_face_train_self.txt"

# 打开文件进行读写操作
with open( input_file , "r" ) as infile , open( output_file , "w" ) as outfile :
	for line in infile :
		# 分割每一行，假设路径是第一部分，后面是其他数据
		parts = line.strip().split( maxsplit = 1 )
		if len( parts ) == 0 :
			continue  # 跳过空行
		
		# 原始路径
		original_path = parts[ 0 ]
		# 其他数据
		other_data = parts[ 1 ] if len( parts ) > 1 else ""
		
		# 修改路径：将原始路径的前缀替换为新目录
		new_path = original_path.replace( "./dataset" , "../dataset" )
		
		# 将修改后的路径和其他数据重新组合成一行
		modified_line = f"{new_path} {other_data}\n"
		
		# 写入修改后的数据
		outfile.write( modified_line )

print( f"数据修改完成，已保存到 {output_file}" )
