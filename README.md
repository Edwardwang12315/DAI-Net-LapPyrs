# 好用的调试代码

### 检查是否有NaN或Inf
```python
# 在数据加载后添加检查
def check_nan_inf(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        print(f"Inf detected in {name}")

check_nan_inf( x , "输入x" )
```
### 把tensor转为图片并保存
```python
print( '暗图' )
image = np.transpose( R_dark[ 0 ].detach().cpu().numpy() , (1 , 2 , 0) )  # 调整维度顺序 [C, H, W] → [H, W, C]
image = (image * 255).astype( np.uint8 )
plt.imshow( image )
plt.axis( 'off' )
# 保存图像到文件
plt.savefig( f'train_暗图.png' , bbox_inches = 'tight' , pad_inches = 0 , dpi = 800 )
```
### 检查GPU内存
```python
print( f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB" )
```
