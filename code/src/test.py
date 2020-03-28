import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--real_time', default=False, type=bool)
opt = parser.parse_args()
print (opt.real_time)


# def split_helper(x,arr):
#     split_arr = []
#     for i in range(len(arr)):
#         if i+x <= len(arr):
#             split_arr.append(arr[i:i+x])
#     return split_arr


# split_arr = split_helper(3,[2,5,4,6,8])
# ans = max([min(subarray) for subarray in split_arr])
# print (ans)

# def split_helper(x,arr):
#     split_arr = []
#     for i in range(len(arr)):
#         if i+x <= len(arr):
#             split_arr.append(min(arr[i:i+x]))
#     return max(split_arr)
# print (split_helper(3,[2,5,4,6,8]))


# def split_helper(x,arr):
#     global_max = 0
#     for i in range(len(arr)-x):
#         if i+x <= len(arr):
#             if i > 0:
#                 #tmp_max = split_arr[-1]
#                 local_min = min(arr[i:i+x])
#                 print ("local_min",local_min)
#                 if local_min > tmp_max:
#                     global_max = local_min
#                     #split_arr.append(local_min)
#             else:
#                 tmp_max = min(arr[i:i+x])
#                 print ("tmp_max", tmp_max)
#                 #split_arr.append(min(arr[i:i+x]))
#     return global_max
#     #return split_arr[-1]
# #ans = split_helper(3,[2,5,4,6,8])
# #print (ans)
# for j in range(0,3):
#     for i in range(0,j):
#         print ("j",j)
#         print ("i",i)
# A recursive function used by countWays 
def countWays(n) : 
    res = [0] * (n + 1) 
    res[0] = 1
    res[1] = 1
    res[2] = 2
      
    for i in range(3, n + 1) : 
        res[i] = res[i - 1] + res[i - 2] + res[i - 3] 
      
    return res[n] 
  
# Driver code 
n = 3
print(countWays(n)) 
n = 4
print(countWays(n)) 
