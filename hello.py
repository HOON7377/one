# name =input("이름을 입력하세요")
# #n_space : number of space(상하좌우 공백갯수)
# n_space= int(input('공백의 수는 얼마인가요 > '))
#
# print('*'*n_space+'*' * 18+'*'*n_space)
#
# for i in range(n_space):
#     print('*'+' '*n_space+' '*16+' '*n_space+'*')
#
# print('*'+' '*n_space+'Hello, Python!!!'+' '*n_space+'*')
#
# for i in range(n_space):
#     print('*'+' '*n_space+' '*16+' '*n_space+'*')
#
# print('*'*n_space+'*' * 18+'*'*n_space) // 길동 공백문제
import math

import conda_verify.utilities
import keyboard
# str ='Hello, Python'
# print(str[-1])
# print(str[1:8])
# 파이썬 문자열 슬라이스 개념과 출력 어디'부터' 어디'전까지'

#
# string1 = "red apple"
# string2 = 'yellow banana'
#
# # red banana 출력
# print(string1[:4]+string2[7:])
# # yellow apple 출력
# print(string2[:7]+string1[4:])



# #Hong Gildong -> Hong
# hong = "Hong Gildong"
# gil = hong
# print(gil[:4])


# str1 = "red apple"
# str2 = "yellow banana"
#
# # #red banana 출력
# # print(str1[:4])

# #키입력하기
# h=int(input("키 입력하시오 > "))
# print("당신의 키는 ", h ,"cm입니다")
# # 변수형이다르면 콤마를 써야함. 강제로 공백생김
# print("당신의 키는 "+str(h)+"cm입니다")

# height = float(input("키를 입력하세요 > "))
#
# print("당신의 키는",height,"cm 입니다.")
# print("당신의 키는 "+str(height)+"cm 입니다.")
# # print : python 2.x버전이후 지원
# print("당신의 키는 %.1fcm입니다."%(height))
# # print문 python 3.0버전 이후 .format
# print("당신의 키는 {0:.1f}cm입니다.".format(height))
# # print문 python 3.6버전 이후 f'
# print(f'당신의 키는 {height:.1f}cm입니다.')




# num=5.94343
# print("%10.2f"%(num))
# print("%10.3f"%(num))
# print("%-10.2f"%(num))

# n1=1
# n2=2
# n3 = n1 + n2
# print("%d + %d = %d"%(n1,n2,n3))
# print("{0:02d} + {1:02d} = {2:02d}".format(n1,n2,n3))
# print(f'{n1:02d} + {n2:02d} = {n3:02d}')

# fav = "meat"
# print("my favorite is %s"%(fav))
# print("my favorite is {}".format(fav))
# print(f"my favorite is {fav}")


# print("Product : %s, Price per uint : %f"%("Apple",5.24))
# print("Product : {}, Price per uint : {}".format("Apple",5.24))
# print(f"Product : {'Apple'}, Price per uint : {5.24}")

# num = 5.94343
# # v2.x
# print("%10.3f"%(num))
# print("%10.2f"%(num))
# print("%-10.2f"%(num))  #왼쪽정렬
# # .format()
# print("{:>10.3f}".format(num))  # > 오른쪽 정렬
# print("{:>10.2f}".format(num))
# print("{:<10.2f}".format(num))  # < 왼쪽정렬
# # f'
# print(f"{num:>10.3f}")  # > 오른쪽 정렬
# print(f"{num:>10.2f}")
# print(f"{num:<10.2f}")  # < 왼쪽정렬


# fruit = 'Apple'
# print("%10s"%fruit)
# print("%-10s"%fruit)
# #.format
# print("{:>10s}".format(fruit)) #오른쪽 정렬은 기본 옵션 : 생략가능
# print("{:<10s}".format(fruit))
# # f'
# print(f"{fruit:>10s}")  #오른쪽 정렬은 기본 옵션 : 생략가능
# print(f"{fruit:<10s}")


# height = int(input("당신의 키는 > "))
# weight = (height-100)*0.9
# print(f'당신의 적정 몸무게는 {int(weight)}kg입니다.')



#퀴즈!!! 231114
# 초를 입력받아 몇시간 몇분 몇초인지 계산
# 시간을 입력하세요(초단위) > 1000
# 1000초는 0시간 16분 40초 입니다.

# s=int(input('계산할 초를 입력 하세요 > '))
# print(f'{s}초는 {s//60//60//24%30}일 {s//60//60%24}시간 {s//60%60}분 {s%60}초 입니다.')


# str1 = 'abc'
# str1 = str1.replace('c','d')
# print(str1)

# shopping_list = ['apple', 'banana', 'orange']
# shopping_list.append('grape')  # 항목 추가
# # shopping_list.remove('apple')  # 항목 삭제
# print(shopping_list)


# ########################### 여기부터 넘파이 배워서 씀.
import numpy as np
#
# A= np.array([1,2,3])
# print(A)
# print(len(A))
# print(type(A))
# print(type(A[0]))
#
# B = np.array([list([1,2,3]),list([4,5])],dtype=object)

# f = np.array([1.1, 2.2, 3.3, 4.9])
# print(f.dtype)
# f = f.astype(np.int32)
# print(f)
# print(f.shape)

# A = [[1,2,3],[4,5,6]]
# A = np.array(A)
# print(A)
# print(type(A))
# print(A.ndim)
# print(A.shape)
# print(A.size)
#
# print(A.itemsize)  # 요소의 바이트 단위사이즈


# b = np.array([1,2,3,4,5,6])
# print('max:',b.max())
# print('min:',b.min())
# print('sum:',b.sum())
# print('mean:',b.mean())


# 방향성 지정후 연산
# axis=0 행계산 / axis=1 열계산

# c =np.array([[1,2],[3,4]])
# print("모든 원소의 합 :",c.sum())
# print(c.sum(axis=0)) # column의 합

# A = np.arange(120)
# print(A)
# A=A.reshape(2,3,4,5)
# print(A)

# import timeit
# print(timeit.timeit('[i**2 for i in A]', setup='A=range(100)'))
# print(timeit.timeit('B**2', setup='import numpy as np;B=np.arange(100)'))

# A= np.zeros((2,3))
# print(A)
#
# B= np.ones((2,3))
# print(B)
#
# C = np.empty((5,5))
# print(C)   # 초기화하지 않기 때문에
# #  대용량 데이터 처리시 속도가 빠르다는 장점

# D = np.random.random((3,3)) # 0~1사이의 실수값
# print(D)          # D = np.random.rand(3,3) 도 같다.
#
#
# # 정수 랜덤값 np.random.randint(시작값,끝값,배열크기(n,m))
# # 실수 랜덤값 np.random.uniform(1,10,(2,2))
# E = np.random.uniform(1,10,(2,2))
# print(E)
# print(type(E))

# # 0에서 부터 100까지의 연속된 정수값
# A = np.arange(101)
# print(A)
#
# # 0부터 50까지 5의 배수값(0포함)
# A = np.arange(0,51,5)
# print(A)

# # 0부터 10까지의 범위를 10개의 숫자 등간격 나누어 배열생성
# A =np.linspace(0,10,10)
# print(A)
#
# B = np.linspace(0,50)  # 디폴트값은 50개
# print(B)
# print(B.size) # B인자 개수

# A = np.arange(16)
# print(A)
#  B=A.reshape(4,4)  # 이렇게하면 B와 A가 값을 함께 "공유"함.어느 한쪽만 바꿔도 둘다 바뀜.
# print(B)
#
# B[0] = 100   # 공유하는지 테스트 한 것.
# print(A)
# print(B)

# B1 = A.reshape(4,4).copy()  # 공유시키지않기위해 B1에는 복사해서 값을 넣어줌.
# print(B1)
#
# B1[0] = 100   # 공유 안하는지 테스트 한 것.
# print(A)
# print(B1)

# B = A.reshape(8,-1)
# print(B)
#
# B = A.reshape(-1,8)
# print(B)

# # ========== 리스트 231115 ===========
# a = list(range(5))
# print(a)
# a = list(range(1,5))
# print(a)
# a = list(range(1,10,3))
# print(a)

# str = 'PYTHON'
# b = list(str)
# print(b)

# # 1부터 100까지 3씩 더해지는 숫자들의 모임
# c = list(range(1,101,3))
# print('[',end=" ")
# for i in range(len(c)):
#     print(c[i], end=" ")
# print(']')

# #1부터 100까지 짝수 모두 출력(100포함)
# for i in range(1,101):
#     if(i%2==0):
#         print(f"{i} ", end="")

# a = list(range(2,101,2))    # 포문안쓰고도 가능.
# print(a)

# # 인덱싱 Indexing
# colors = ['red','blue','yellow']
# print(len(colors))
#
# for i in range(len(colors)):
#     print(colors[i])
#
# for color in colors:
#     print(color, end=" ")

# a = [10, 20, 30]
# a.insert(len(a), 500)
# print(a)

# a = [10, 20, 30]
# a[1:1] = [500, 600]
# print(a)

# a = [10, 20, 30]
# a.insert(len(a), 500)
# print(a)

# a = [10, 20, 30]
# a.extend([500, 600])
# print(a)
# print(len(a))


# fruits = ['mango','grape','cherry']
# foods = ['pizza','salad', 'stake']
# # fruits.extend(foods)
# mixed = fruits + foods
# mixed[len(fruits)-1] = 'kiwi'
# print(mixed)

# fruits = ['mango','grape','cherry']
# foods = ['pizza','salad', 'stake']
# # fruits.extend(foods)
# removed_fruit=fruits.pop()
# fruits[fruits.index(removed_fruit)]='kiwi'
#
# mixed = fruits + foods
#
# print(mixed)


# hero = ['아이언맨', '토르', '헐크', '스칼렛위치']
# hero[hero.index('토르')] ='닥터스트레인지'
# print(hero)

# =================231116====================

# B = np.array([[1, 2, 3, 4],
#               [5, 6, 7, 8],
#               [9, 10, 11, 12]])
#
# b = np.hsplit(B, 2)
#
# print(b)

# A = np.arange(6).reshape(2,3)
# B= np.arange(4).reshape(2,2)
# C = np.hstack((A,B))
# print(C)

# A = np.arange(0,15,2)
# length = len(A) # length = 8
# print(A[0:length-2])
# print(A[:-2])

# a = np.arange(12)
# c = a.reshape(3,4)
# c[0]=5
# print(a)
# print(c)
# print(id(a))
# print(id(c.base))


# ===============231116 집합 ==========
# s = set()
# n = np.arange(1,46)
# print(f"n출력{n}",end=' ')
# print("")
# for i in n:
#     s.add(i)
#
# print(f"s출력{s}",end=' ')
# print("")

# x1 =set()
# y=set()
# y.add(np.random.randint(1,46))
# print(f"당첨번호{y}")
# i=1
# while (i<6) :
#     while (len(x1)!=6) :
#         x1.add(np.random.randint(1,46))
#     list_1=[]
#     list_1+=x1
#     list_1.sort()
#     # k=set()
#     # k=(x1&y)   # x1과 y의 교집합을 k에 추가한다.
#     # # print(x1&y)
#     # # print(k)
#
#     print(f"{i}번째 응모 번호 {list_1}",end="\t ")
#     print(f"{i}번째 게임 당첨갯수 : {len(x1&y)}")
#     x1=set()
#     i+=1

# card_type = ['♥','♣','♠','◆']
# card_num = ['1','2','3','4','5','6','7','8','9','10','J','Q','K']
# c=set()
# for a in card_type:
#     for b in card_num:
#         c.add(a+b)
#
# while (True) :
#     i=input("카드를 뽑으세요 (나가려면 q를 누르세요) > ")
#     d = c.pop()
#
#     print(f"뽑은카드는 {d}입니다.")
#     if (i=='q') :
#         break


# k= np.cos(np.pi / 4)
# print(f"{k:.15f}")
# print(f"{1/np.sqrt(2):.30f}")

# ===================231117 오픈cv=======
import cv2

# 이미지 출력
# img_src = cv2.imread('s1213.jpg',cv2.IMREAD_COLOR)
# height,width = img_src.shape[:2]
# img_src=cv2.pyrDown(img_src )
# img_src=cv2.pyrDown(img_src )
# cv2.imshow('src', img_src)
# cv2.waitKey()
# cv2.destroyWindow()


# # 45도 사진회전하기
# import cv2
# import numpy as np
#
# img = cv2.imread('love.png', cv2.IMREAD_COLOR)
# height, width = img.shape[:2]
#
# # 45도를 라디안으로 변환하여 코싸인값과 싸인값을 구합니다.
# angle = 45
# radian = angle * np.pi / 180
# c = np.cos(radian)
# s = np.sin(radian)
#
# # 회전변환행렬을 구성합니다.
# # OpenCV의 원점이 왼쪽아래가 아니라 왼쪽위라서 [[c, -s, 0], [s, c, 0]]가 아니라
# # [[c, s, 0], [-s, c, 0]]입니다.
# rotation_matrix = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=float)
#
# dst = np.zeros(img.shape, dtype=np.uint8)
#
# for y in range(height - 1):
#     for x in range(width - 1):
#
#         # backward mapping
#         # 결과 이미지의 픽셀 new_p로 이동하는 입력 이미지의
#         # 픽셀 old_p의 위치를 계산합니다.
#         new_p = np.array([x, y, 1])
#         inv_rotation_matrix = np.linalg.inv(rotation_matrix)
#         old_p = np.dot(inv_rotation_matrix, new_p)
#
#         # new_p 위치에 계산하여 얻은 old_p 픽셀의 값을 대입합니다.
#         x_, y_ = old_p[:2]
#         x_ = int(x_)
#         y_ = int(y_)
#
#         # 입력 이미지의 픽셀을 가져올 수 있는 경우에만
#         # 결과 이미지의 현재 위치의 픽셀로 사용합니다.
#         if x_ > 0 and x_ < width and y_ > 0 and y_ < height:
#             dst[y, x] = img[y_, x_]
#
# result = cv2.hconcat([img, dst])
# cv2.imshow("result", result)
# cv2.waitKey(0)

# ===============231121 이미지 축소(종합세트) ==================
# import cv2
# import numpy as np
#
# img = cv2.imread('love.png', cv2.IMREAD_COLOR)
# height, width = img.shape[:2]
#
# scale_factor = 0.1
# scaling_matrix = np.array([[scale_factor, 0, 0], [0, scale_factor, 0], [0, 0, 1]])
# translation_matrix = np.array([[1, 0, width*(1-scale_factor)/2], [0, 1,height*(1-scale_factor)/2], [0, 0, 1]])
# angle = 45
# radian = angle * np.pi / 180
# c = np.cos(radian)
# s = np.sin(radian)
# center_x = width / 2
# center_y = height / 2
# rotation_matrix = np.array(
#     [[c, s, (1 - c) * center_x - s * center_y], [-s, c, s * center_x + (1 - c) * center_y], [0, 0, 1]])
#
# # 정해진 순서대로 변환 행렬을 곱하여 하나의 행렬을 생성합니다.
# T = np.eye(3)
# T = np.dot(scaling_matrix, T)
# T = np.dot(translation_matrix, T)
# T = np.dot(rotation_matrix, T)
#
# dst = np.zeros((height, width, img.shape[2]), dtype=np.uint8)
#
# for y in range(height):
#     for x in range(width):
#
#         # 미리 구해놓은 변환행렬을 행렬곱 한번으로 적용합니다.
#         # 여기에서도 backward mapping을 사용합니다.
#         new_p = np.array([x, y, 1])
#         inv_scaling_matrix = np.linalg.inv(T)
#         old_p = np.dot(inv_scaling_matrix, new_p)
#
#         x_, y_ = old_p[:2]
#         x_ = int(x_)
#         y_ = int(y_)
#
#         if x_ > 0 and x_ < width and y_ > 0 and y_ < height:
#             dst[y, x] = img[y_, x_]
#
# cv2.imshow("result", dst)
# cv2.waitKey(0)


# import cv2
# import numpy as np
#
# # 클릭한 횟수를 저장합니다.
# count_mouse_click = 0
#
# # 호모그래피 행렬을 곱해서 결과를 계산중이면 1을 갖게됩니다.
# # 계산중에 마우스 클릭은 무시하기 위해서 사용됩니다.
# caculate_start = 0
#
# # 마우스 클릭한 위치를 저장할 리스트입니다.
# pointX = []
# pointY = []
#
#
# # OpenCV 창에 보이는 이미지를 클릭시, 클릭한 위치(x,y)를 파라미터로 호출되는 콜백함수입니다.
# def CallBackFunc(event, x, y, flags, userdata):
#     global count_mouse_click, caculate_start
#
#     # 마우스 왼쪽 버튼을 클릭했는지 체크합니다.
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print("{} - ({}, {} )".format(count_mouse_click, x, y))
#
#         # 마우스 클릭한 위치를 저장합니다.
#         pointX.append(x)
#         pointY.append(y)
#
#         # 마우스 클릭한 횟수를 업데이트합니다.
#         count_mouse_click += 1
#
#         # 마우스 클릭한 위치를 화면에 보여줄 때 사용하기 위해 입력 이미지를 복사합니다.
#         img_temp = img_gray.copy()
#
#         # 마우스 클릭한 위치에 원을 그립니다.
#         for point in zip(pointX, pointY):
#             cv2.circle(img_temp, point, 5, (0), 2)
#
#         # 마우스 클릭할때마다 원이 이미지에 보이게 됩니다.
#         cv2.imshow("gray image", img_temp)
#
#     # 4점을 모두 클릭한 상태이고 아직 결과 이미지를 처리하기 전이면
#     if count_mouse_click == 4 and caculate_start == 0:
#
#         # 이제 결과 이미지 처리중임을 알립니다.
#         caculate_start = 1;
#
#         print("calculate H")
#
#         # 클릭한 사각 영역좌표를 기반으로 정면에서 바라본 직사각형 영역을 계산합니다.
#         width = ((pointX[1] - pointX[0]) + (pointX[3] - pointX[2])) * 0.5;
#         height = ((pointY[2] - pointY[0]) + (pointY[3] - pointY[1])) * 0.5;
#
#         newpointX = np.array([pointX[3] - width, pointX[3], pointX[3] - width, pointX[3]])
#         newpointY = np.array([pointY[3] - height, pointY[3] - height, pointY[3], pointY[3]])
#
#         # 계산한 직사각형 영역을 화면에 출력합니다.
#         for i in range(4):
#             print("({}, {})".format(newpointX[i], newpointY[i]))
#
#         # 마우스로 클릭한 좌표와 계산된 좌표를 넘파이 배열로 변환합니다.
#         pts_src = []
#         pts_dst = []
#
#         for i in range(4):
#             pts_src.append((pointX[i], pointY[i]))
#             pts_dst.append((newpointX[i], newpointY[i]))
#
#         pts_src = np.array(pts_src)
#         pts_dst = np.array(pts_dst)
#
#         # 호모그래피 행렬을 구합니다.
#         A = np.array([
#             [-1 * pointX[0], -1 * pointY[0], -1, 0, 0, 0, pointX[0] * newpointX[0], pointY[0] * newpointX[0],
#              newpointX[0]],
#             [0, 0, 0, -1 * pointX[0], -1 * pointY[0], -1, pointX[0] * newpointY[0], pointY[0] * newpointY[0],
#              newpointY[0]],
#             [-1 * pointX[1], -1 * pointY[1], -1, 0, 0, 0, pointX[1] * newpointX[1], pointY[1] * newpointX[1],
#              newpointX[1]],
#             [0, 0, 0, -1 * pointX[1], -1 * pointY[1], -1, pointX[1] * newpointY[1], pointY[1] * newpointY[1],
#              newpointY[1]],
#             [-1 * pointX[2], -1 * pointY[2], -1, 0, 0, 0, pointX[2] * newpointX[2], pointY[2] * newpointX[2],
#              newpointX[2]],
#             [0, 0, 0, -1 * pointX[2], -1 * pointY[2], -1, pointX[2] * newpointY[2], pointY[2] * newpointY[2],
#              newpointY[2]],
#             [-1 * pointX[3], -1 * pointY[3], -1, 0, 0, 0, pointX[3] * newpointX[3], pointY[3] * newpointX[3],
#              newpointX[3]],
#             [0, 0, 0, -1 * pointX[3], -1 * pointY[3], -1, pointX[3] * newpointY[3], pointY[3] * newpointY[3],
#              newpointY[3]]])
#
#         u, s, v = np.linalg.svd(A, full_matrices=True)
#         v = v.T
#
#         # v의 마지막 컬럼값을 H로 취합니다.
#         temp = v[:, 8]
#         h = temp.reshape(3, 3)
#
#         # h_33을 1로 만듭니다.
#         h = h / h[2, 2]
#
#         img_result = np.zeros(img_gray.shape, dtype=np.uint8)
#
#         inv_h = np.linalg.inv(h)
#
#         height, width = img_gray.shape[:2]
#         for y in range(height):
#             for x in range(width):
#
#                 # 변환 후 좌표를 기준으로 원본 이미지상의 좌표 계산
#                 newpoint = np.array([x, y, 1])
#                 oldpoint = np.dot(inv_h, newpoint)
#
#                 oldX = int(oldpoint[0] / oldpoint[2])
#                 oldY = int(oldpoint[1] / oldpoint[2])
#
#                 # 원본 이미지의 좌표상의 픽셀을 현재 위치로 가져옴
#                 if oldX > 0 and oldY > 0 and oldX < width and oldY < height:
#                     img_result.itemset(y, x, img_gray.item(oldY, oldX))
#
#         result = cv2.hconcat([img_gray, img_result])
#         cv2.imshow("result", result)
#         cv2.waitKey(0)
#
#
# # 호모그래피 행렬을 저장할 입력 이미지를 로드합니다.
# img_gray = cv2.imread("wwww.jpg", cv2.IMREAD_GRAYSCALE)
#
# # 타이틀바에 “gray image”를 출력하는 창에 넘파이 배열 img_gray를 보여줍니다.
# cv2.imshow("gray image", img_gray)
#
# # 타이틀바에 “gray image”를 출력하는 창을 위해 마우스 콜백 함수를 지정합니다.
# cv2.setMouseCallback("gray image", CallBackFunc)
#
# print("left up, right up, left down, right down")
#
# cv2.waitKey(0)
#

# import cv2
# import numpy as np
#
# # 이미지 불러오기
# img = cv2.imread('love.png', cv2.IMREAD_COLOR)
# height, width = img.shape[:2]
#
# # 변환 행렬 생성
# scale_factor = 2
# scaling_matrix = np.array([[scale_factor, 0, 0], [0, scale_factor, 0], [0, 0, 1]])
# translation_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# angle = 45
# radian = angle * np.pi / 180
# c = np.cos(radian)
# s = np.sin(radian)
# center_x = width / 2
# center_y = height / 2
# rotation_matrix = np.array([[c, s, (1 - c) * center_x - s * center_y], [-s, c, s * center_x + (1 - c) * center_y], [0, 0, 1]])
# T = np.eye(3)
# T = np.dot(scaling_matrix, T)
# T = np.dot(translation_matrix, T)
# T = np.dot(rotation_matrix, T)
#
# # 색상 채널을 나누고 변환 적용
# dst_split = np.zeros((height, width, img.shape[2]), dtype=np.uint8)
#
# for y in range(height):
#     for x in range(width):
#         new_p = np.array([x, y, 1])
#         inv_scaling_matrix = np.linalg.inv(T)
#         old_p = np.dot(inv_scaling_matrix, new_p)
#
#         x_, y_ = old_p[:2]
#         x_ = int(x_)
#         y_ = int(y_)
#
#         if 0 < x_ < width and 0 < y_ < height:
#             dst_split[y, x] = img[y_, x_]
#
# # 변환 행렬을 직접 적용
# dst_direct = np.zeros_like(img)
#
# for y in range(height):
#     for x in range(width):
#         new_p = np.array([x, y, 1])
#         inv_scaling_matrix = np.linalg.inv(T)
#         old_p = np.dot(inv_scaling_matrix, new_p)
#
#         x_, y_ = old_p[:2]
#         x_ = int(x_)
#         y_ = int(y_)
#
#         if 0 < x_ < width and 0 < y_ < height:
#             dst_direct[y, x] = img[y_, x_]
#
# # 결과 표시
# cv2.imshow("Original Image", img)
# cv2.imshow("Transformed with Channel Splitting", dst_split)
# cv2.imshow("Transformed without Channel Splitting", dst_direct)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import random
# import keyboard
# print("주사위 프로그램을 시작합니다")
#
# while True :
#     print("아무키나 누르면 주사위가 던져집니다. 종료를 원하시면  'q'를 입력해주세요.")
#     if keyboard.read_key() =='q':
#         print('종료합니다.')
#         break
#     else:
#         print(f'\n{random.randint(1,6)}')

# # 리스트 컴플리케이션?
# ns=[1,2,3,4,5]
# resurt = [n*2 for n in ns if n%2==1 ]
# print(resurt)

# n= int(input('원하는숫자입력 > '))
# resum = 0
# resum = sum([n for n in range(1,n+1) if True])
# print(f"1부터 {n}까지 합 :{resum}")


# n= int(input('원하는숫자입력 > '))
# Sum=0
# i=0
# while i<=n:
#     Sum+=i
#     i+=1
# print(f"합 : {Sum}")

# import cv2
# # source : img_src // destination : img_dst
# # 이미지 읽어들이기
# img_src = cv2.imread('love.png', cv2.IMREAD_COLOR)
# height,width = img_src.shape[:2]
#
#
# #========================== 이미지를 화면에 출력
# cv2.imshow('src', img_src)
# cv2.waitKey()
# cv2.destroyAllWindows()

# # 동영상
# import cv2
#
# capture=cv2.VideoCapture("lookbook.mp4")
# if capture.isOpened() == False:
#     print("동영상을 열 수 없음")
#     exit(1)
#
# while cv2.waitKey(3)<0:
#     ret,img_frame =capture.read()
#     if ret ==False:
#         print("캡쳐실패")
#         break
#
#     cv2.imshow('Color',img_frame)
#
#     key=cv2.waitKey(3)
#     if key==30:
#         break
# capture.release()
# cv2.destroyAllWindows()

# =========================이미지(그레이도 출력)==============
# import cv2
#
# # 이미지로딩
# img_src = cv2.imread('love.png',cv2.IMREAD_COLOR)
# height,width = img_src.shape[:2]
#
# # 이미지를 칼라에서 그레이로 바꾼다
# img_gray = cv2.cvtColor(img_src,cv2.COLOR_BGR2GRAY)
#
# #결과출력
# cv2.imshow('src', img_src)
# cv2.imshow('dst', img_gray)
# cv2.waitKey()
# cv2.destroyAllWindows()

# =================동영상 그레이도 출력 ==========
# # 동영상
# import cv2
#
# capture=cv2.VideoCapture("lookbook.mp4")
# if capture.isOpened() == False:
#     print("동영상을 열 수 없음")
#     exit(1)
#
# while cv2.waitKey(3)<0:
#     ret,img_frame =capture.read()
#
#     img_gray = cv2.cvtColor(img_frame,cv2.COLOR_BGR2GRAY)
#
#     if ret ==False:
#         print("캡쳐실패")
#         break
#
#     cv2.imshow('Color',img_gray)
#
#     key=cv2.waitKey(3)
#     if key==27:  # 27=esc아스키코드
#         break
# capture.release()
# cv2.destroyAllWindows()


# ==============그레이영상을 옆에 축소 해서 같이출력 =========
# import cv2
#
# capture = cv2.VideoCapture('eta.mp4')
# if capture.isOpened() == False:
#     print("동영상을 열수 없음")
#     exit(1)
#
# while cv2.waitKey(25) < 0: #아무키나 누르면 종료
#     ret, img_src = capture.read()
#     img_src = cv2.pyrDown(img_src)
#     img_gray = cv2.cvtColor(img_src,cv2.COLOR_BGR2GRAY)
#     img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
#     img_dst = cv2.vconcat([img_src,img_gray])
#     # 동영상 끝까지 읽으면 처음으로 돌아감
#     if capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT):
#         capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
#     cv2.imshow('dst', img_dst)
# capture.release()
# cv2.destroyAllWindows()


# ================이미지 대칭으로 출력하기 =========
# import cv2
#
# img_src = cv2.imread('love.png',cv2.IMREAD_COLOR)
# height, width = img_src.shape[:2]
# # 대칭옵션 : 0=상하 / 양수=좌우 / 음수=상하좌우
# img_lr = cv2.flip(img_src,1)
# img_ud = cv2.flip(img_src,0)
# img_lr_ud = cv2.flip(img_src,-1)
# img_dst = cv2.hconcat([img_src,img_lr])
# img_tmp = cv2.hconcat([img_ud,img_lr_ud])
# img_dst = cv2.vconcat([img_dst,img_tmp])
#
# cv2.imshow('dst',img_dst)
# cv2.waitKey()
# cv2.destroyAllWindows()


# ===========동영상 축소시킨거 대칭으로뽑기=======
# import cv2
#
# capture = cv2.VideoCapture('lookbook.mp4')
# if capture.isOpened() == False:
#     print("동영상을 열수 없음")
#     exit(1)
#
# while cv2.waitKey(10) ==-1: #아무키나 누르면 종료
#     ret, img_src = capture.read()
#     img_src = cv2.pyrDown(img_src)
#     img_lr =cv2.flip(img_src,1)
#     img_ud = cv2.flip(img_src, 0)
#     img_lr_ud = cv2.flip(img_src, -1)
#
#     img_dst = cv2.hconcat([img_src, img_lr])
#     img_temp= cv2.hconcat([img_ud, img_lr_ud])
#     img_dst = cv2.vconcat([img_dst, img_temp])
#
#     if ret == False:
#         print("캡쳐실패")
#         break
#     cv2.imshow('dst', img_dst)
#     key=cv2.waitKey(10)
#     if key==27:
#         break
#
# capture.release()
# cv2.destroyAllWindows()


# =============이미지 회전 ============
# import cv2
# img_src = cv2.imread('love.png',cv2.IMREAD_COLOR)
# height,width = img_src.shape[:2]
# center = (width//2,height//2)
# angle = 10
# matrix=cv2.getRotationMatrix2D(center,angle,1)
# img_dst = cv2.warpAffine(img_src,matrix,(width,height))
#
# cv2.imshow('src',img_dst)
# cv2.waitKey()
# cv2.destroyAllWindows()


# ==========이미지 축소해보기==============
# import cv2
#
# img_src= cv2.imread('love.png',cv2.IMREAD_COLOR)
#
#
# img_dst=cv2.pyrDown(img_src)    #dst에 1/2배율사진 생성
# height,width = img_dst.shape[:2] #h와 w 크기 정의.
# img_dst2=cv2.pyrDown(img_dst)    #dst2에 1/4배율사진 생성
#
#
# img_tmp=np.zeros_like(img_src)  # tmp에 원본크기 검정화면 생성
# img_tmp2=img_tmp.copy()  # tmp2에 원본크기 검정화면 생성
#
#
#
# img_tmp[:height,:width,:] = img_dst
#     # 검정화면에 1/2배율 사진의 크기를 :h:w영역에만 복붙
# img_dst = cv2.hconcat([img_src,img_tmp])
#
# height,width = img_dst2.shape[:2] #h와 w 크기 정의.
#
# img_tmp2 [:height,:width,] = img_dst2
#
# img_dst = cv2.hconcat([img_dst,img_tmp2])
#
#
#
#
# cv2.imshow('src', img_dst)
# cv2.waitKey()
# cv2.destroyAllWindows()


# # =============동영상 축소하기 ========================================
# # video 1/2크기로 변환후 hconcat
# capture = cv2.VideoCapture('lookbook.mp4')
# if capture.isOpened() == False:
#     print("동영상을 열수 없음")
#     exit(1)
#
#
# while cv2.waitKey(33) < 0: #아무키나 누르면 종료
#     ret, img_src = capture.read()
#     img_src = cv2.pyrDown(img_src)  # src는 1/2배율 영상
#     img_dst = np.zeros_like(img_src)
#     img_tmp = cv2.pyrDown(img_src)  # tmp 는 1/4배율 영상
#     h,w = img_tmp.shape[:2]
#     img_dst[h//2:3*h//2,w//2:3*w//2,:] = img_tmp
#     img_dst = cv2.hconcat([img_src,img_dst])
#     # 동영상 끝까지 읽으면 처음으로 돌아감
#     if capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT):
#         capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
#     cv2.imshow('dst', img_dst)
# capture.release()
# cv2.destroyAllWindows()



# ================이미지자르기 ===========
# import cv2
#
# img_src = cv2.imread('love.png',cv2.IMREAD_COLOR)
# img_crop = img_src[176:530,73:290,:]
#
# img_crop = cv2.cvtColor(img_crop,cv2.COLOR_BGR2GRAY) # 1ch gray
# img_crop = cv2.cvtColor(img_crop,cv2.COLOR_GRAY2BGR) # 3ch gray
#
# img_src[176:530,73:290,:] = img_crop
# cv2.imshow('src',img_src)
# cv2.waitKey()
# cv2.destroyAllWindows()

# =========칼라안에 흑백과  흑백안에 칼라.  영상 만들기 ====
# video 1/2크기로 변환후 hconcat
# import cv2
#
# capture = cv2.VideoCapture('lookbook.mp4')
# if capture.isOpened() == False:
#     print("동영상을 열수 없음")
#     exit(1)
#
#
# while cv2.waitKey(33) < 0: #아무키나 누르면 종료
#     ret, img_src = capture.read()
#     img_src = cv2.pyrDown(img_src)  # 너무커서 1/2배율로 줄이고 시작함.
#     h,w = img_src.shape[:2]
#     img_crop2 = img_src.copy() #큰사진을 전체 크롭으로 복사함
#
#     img_crop2 = cv2.cvtColor(img_crop2, cv2.COLOR_BGR2GRAY)  # 1ch gray
#     img_crop2 = cv2.cvtColor(img_crop2, cv2.COLOR_GRAY2BGR)  # 3ch gray
#     # 칼라원본을 복사해온 crop2를 전체 흑백으로 바꾼다.
#
#     img_src2 = img_src[h//4:3*h//4,w//4:3*w//4,:].copy()
#     img_crop2[h//4:3*h//4,w//4:3*w//4,:]=img_src2
#
#     img_crop = img_src[h//4:3*h//4,w//4:3*w//4,:]
#     img_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)  # 1ch gray
#     img_crop = cv2.cvtColor(img_crop,cv2.COLOR_GRAY2BGR) # 3ch gray
#     h,w = img_crop.shape[:2]
#     img_src[h//2:3*h//2,w//2:3*w//2,:] = img_crop  # 칼라안에 흑백 완료.
#
#
#     img_dst = cv2.vconcat([img_src,img_crop2])
#     # 동영상 끝까지 읽으면 처음으로 돌아감
#     if capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT):
#         capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
#     cv2.imshow('dst', img_dst)
# capture.release()
# cv2.destroyAllWindows()


# ===============231123 rgb =============
# import cv2
# import numpy as np
#
#
# img_src =cv2.imread('rgb.png',cv2.IMREAD_COLOR)
# h,w=img_src.shape[:2]
# img_b,img_g, img_r = cv2.split(img_src)
# img_zero = np.zeros((h,w,1),dtype=np.uint8)
# img_3ch_b = cv2.merge((img_b,img_zero,img_zero))
# img_3ch_g = cv2.merge((img_zero,img_g,img_zero))
# img_3ch_r = cv2.merge((img_zero,img_zero,img_r))
#
# img_dst = cv2.merge((img_b, img_g, img_r))
#
# cv2.imshow('src', img_src)
# cv2.imshow('b', img_3ch_b)
# cv2.imshow('g', img_3ch_g)
# cv2.imshow('r', img_3ch_r)
# cv2.imshow('dst', img_dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# =======블러처리 노이즈제거 =======
# import cv2
#
# img_src= cv2.imread('opencv.png',cv2.IMREAD_COLOR)
# img_src=cv2.pyrDown(img_src)
# img_dst= cv2.GaussianBlur(img_src,(15,15),0)
# h,w=img_src.shape[:2]
#
# cv2.imshow('src',img_src)
# cv2.imshow('dst',img_dst)
# cv2.waitKey()
# cv2.destroyAllWindows()


# ==========가우시안블러 스도쿠 사진==============
# import cv2
#
# img_src = cv2.imread('sudoku-original.jpg',cv2.IMREAD_COLOR)
# img_gray = cv2.cvtColor(img_src,cv2.COLOR_BGR2GRAY)
# img_gray = cv2.GaussianBlur(img_gray,(3,3),0)
# # ret, img_dst = cv2.threshold(img_dst,200, 255, cv2.THRESH_OTSU)
# img_dst1 = cv2.adaptiveThreshold(img_gray,255,
#                                 cv2.ADAPTIVE_THRESH_MEAN_C,
#                                 cv2.THRESH_BINARY,11,2)
# img_dst2 = cv2.adaptiveThreshold(img_gray,255,
#                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                 cv2.THRESH_BINARY,11,2)
#
# cv2.imshow('src',img_src)
# cv2.imshow('mean c',img_dst1)
# cv2.imshow('gaussian c',img_dst2)
# cv2.waitKey()


# ===================HSV 색상계로 바꾸기 ==========
# import cv2
# img_src = cv2.imread('tomato.png',cv2.IMREAD_COLOR)
# height, width = img_src.shape[:2]
#
# img_hsv = cv2.cvtColor(img_src,cv2.COLOR_BGR2HSV)
# img_h, img_s, img_v = cv2.split(img_hsv)
# orange_mask = cv2.inRange(img_h, 8,20)  # h채널에서만 범위를 지정해줌.
# img_orange_hsv = cv2.bitwise_and(img_hsv,img_hsv,mask=orange_mask)
# img_orange_bgr = cv2.cvtColor(img_orange_hsv,cv2.COLOR_HSV2BGR)
#
#
# cv2.imshow('src',img_orange_bgr)
# cv2.imshow('h',orange_mask)
#
# cv2.waitKey()
# cv2.destroyAllWindows()
# # =============HSV 전부 값 설정하기
# import cv2
# img_src = cv2.imread('tomato.png',cv2.IMREAD_COLOR)
# height, width = img_src.shape[:2]
#
# img_hsv = cv2.cvtColor(img_src,cv2.COLOR_BGR2HSV)
# h_min = 8; h_max = 20
# s_min = 0; s_max = 255
# v_min = 0; v_max = 255
# orange_mask = cv2.inRange(img_hsv, (h_min,s_min,v_min),(h_max,s_max,v_max))
# img_orange = cv2.bitwise_and(img_hsv,img_hsv,mask=orange_mask)
# img_orange = cv2.cvtColor(img_orange,cv2.COLOR_HSV2BGR)
#
#
# cv2.imshow('src',img_orange)
# cv2.imshow('h',orange_mask)
#
# cv2.waitKey()
# cv2.destroyAllWindows()

# =================영상에서 ktx파란색만 뽑아내기 ======
# import cv2
#
# capture= cv2.VideoCapture('KTX_departing.mp4')
# if capture.isOpened()==False:
#     print('동영상을 열 수 없음')
#     exit(1)
#
# # 관심 영역의 꼭지점 좌표 설정 (p1, p2, p3, p4)
# p1 = (301, 358)
# p2 = (307, 431)
# p3 = (1440, 800)
# p4 = (1440, 82)
#
#
# while cv2.waitKey(1)<0:
#     ret, img_src=capture.read()
#     h,w=img_src.shape[:2]
#     img_hsv = cv2.cvtColor(img_src,cv2.COLOR_BGR2HSV)
#     h_min = 100; h_max = 180
#     s_min = 50; s_max = 240
#     v_min = 30; v_max = 200
#     blue_mask = cv2.inRange(img_hsv, (h_min*180/240, s_min*255/240, v_min*255/240), (h_max*180/240, s_max*255/240, v_max*255/240))
#     img_blue = cv2.bitwise_and(img_hsv,img_hsv,mask=blue_mask)
#     img_blue = cv2.cvtColor(img_blue,cv2.COLOR_HSV2BGR)
#
#     cv2.imshow('h',img_blue)
#     if cv2.waitKey(1)==27:
#         break
#
# capture.release()
# cv2.destroyAllWindows()


# ====231124   원 만들기 ========================
# import cv2
# import numpy as np
# height, width, dim = (600,800,3)
# img_src = np.zeros((height,width,dim),dtype=np.uint8)
# color=(0,255,0)
# cv2.circle((img_src),(300,300),3,(0,0,255),-1)
# cv2.circle((img_src),(300,300),50,(0,255,0),2)
# cv2.circle((img_src),(200,200),100,(0,255,255),-1)
# cv2.circle((img_src),(200,200),3,(0,0,255),-1)
#
# cv2.imshow('src',img_src)
# cv2.waitKey()
# cv2.destroyAllWindows()

# ================================ 원그리기 색깔 랜덤으로 정하기+ 사각형스리키 색깔 랜덤
# import cv2
# import numpy as np
# import random
#
# height, width, dim = (600,800,3)
# img_src = np.zeros((height,width,dim),dtype=np.uint8)
# b = random.randint(0,255)  # 랜드인트함수는 0~255까지 나옴
# g = random.randint(0,255)
# r = random.randint(0,255)
#
#
# cv2.circle((img_src),(600,300),3,(0,0,255),-1)
# cv2.circle((img_src),(600,300),50,(0,255,0),2)
# cv2.circle((img_src),(600,120),100,(b,g,r),-1)
# cv2.circle((img_src),(600,120),3,(0,0,255),-1)
# # cv2.circle((배경이미지),(센터좌표 x , y) ,반지름크기,(0,0,255), 선굵기(음수는 채우기) )
#
# # cv2.imshow('src',img_src)
# # cv2.waitKey()
# # cv2.destroyAllWindows()
#
#
# # 여기부터 사각형 그리기
# cv2.rectangle(img_src,(50,50),(450,450),(0,255,0),3)  # 초록테두리
# cv2.rectangle(img_src,(50,50),(450,450),(0,255,0),3) # 파란사각형
#
# cv2.rectangle(img_src,(150,200),(250,300),(b,g,r),-1)
#                 # 시작점좌표x,y 끝점좌표x,y
# cv2.rectangle(img_src,(300,150, 50, 100),(255,255,0),-1)
#             # ([시작점좌표x,y], 사각형의너비x, 사각형의높이y)
#
# cv2.imshow('src',img_src)
# cv2.waitKey()
# cv2.destroyAllWindows()

# ==========================================선그리기============
# import cv2
# import numpy as np
# import random
#
# height, width, dim = (600,800,3)
# img_src = np.zeros((height,width,dim),dtype=np.uint8)
#
#
# b = random.randint(0,255)  # 랜드인트함수는 0~255까지 나옴
# g = random.randint(0,255)
# r = random.randint(0,255)
#
# cv2.line(img_src,(100,100),(700,100),(0,0,255),5)
# cv2.line(img_src,(100,200),(700,200),(0,255,255),5)
# cv2.line(img_src,(100,300),(700,300),(0,255,0),5)
# cv2.line(img_src,(100,400),(700,400),(255,255,0),5)
# cv2.line(img_src,(100,500),(700,500),(255,0,0),5)
#
# # for y in range(100,600,100):
# #     b = random.randint(0, 255)  # 랜드인트함수는 0~255까지 나옴
# #     g = random.randint(0, 255)
# #     r = random.randint(0, 255)
# #
# #     cv2.line(img_src,(100,y),(700,y),(b,g,r),5)
#
# # 대각선으로 X모양 노란색으로 귿기
# cv2.line(img_src,(0,0),(799,599),(0,255,255),3)
# cv2.line(img_src,(799,0),(0,599),(0,255,255),3)
#
#
#
# cv2.imshow('src',img_src)
# cv2.waitKey()
# cv2.destroyAllWindows()

#==============================타원그리기 ==========================
# import cv2
# import numpy as np
# import random
#
# h,w,d=(600,800,3)
# img_src=  np.zeros((h,w,d),dtype=np.uint8)
# center = (400,300)
# axes = (150,50)
# angle =30
# startAngle = 20
# endAngle = 350
# color = (255,255,0)
# thick = -1
# cv2.ellipse(img_src,center,axes,angle,startAngle,endAngle,color,thick)
#
#
# cv2.imshow('src',img_src)
# cv2.waitKey()
# cv2.destroyAllWindows()

# =================================== 삼각형 그리기+ 텍스트 글자 넣기========
# import cv2
# import numpy as np
# import random
#
# h,w,d=(600,800,3)
# img_src=  np.zeros((h,w,d),dtype=np.uint8)
#
# pt1 = np.array([[100,500],[300,500],[200,550]])
# pt2 = np.array([[600,500],[700,500],[650,550]])
#
# img_src = cv2.polylines(img_src, [pt1], True, (0,255,255),2)
# img_src = cv2.fillPoly(img_src, [pt2], (0,255,255),cv2.LINE_AA)
#
# pts = np.array([[315, 50], [570, 240], [475, 550],
#     [150, 550], [50, 240]])
# pts = pts.reshape((-1, 1, 2))
# cv2.polylines(img_src, [pts], False, (0,0,255), 3)
#
#
#
# cv2.line(img_src,(315,50),(330,50),(0,255,255),1)
# # 글자위치 시작점 확인용 노란선
#
# # 글자 넣기
# location = (315, 50)
# font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
# fontScale = 0.5
# color=(255,255,255)
# thickness= 1
# cv2.putText(img_src,'p1',pts[0][0],font,fontScale,color,thickness)
# cv2.putText(img_src,'p2',pts[1][0],font,fontScale,color,thickness)
# cv2.putText(img_src,'p3',pts[2][0],font,fontScale,color,thickness)
# cv2.putText(img_src,'p4',pts[3][0],font,fontScale,color,thickness)
# cv2.putText(img_src,'p5',pts[4][0],font,fontScale,color,thickness)
#
#
# cv2.imshow('src',img_src)
# cv2.waitKey()
# cv2.destroyAllWindows()

# ============================== 내 그림에 쓰레쓰홀드 적용해보기 =======
# import cv2
# img_src = cv2.imread('Myshape.png',cv2.IMREAD_COLOR)
# h,w=img_src.shape[:2]
# img_gray = cv2.cvtColor(img_src,cv2.COLOR_BGR2GRAY)
# retval, img_gray=cv2.threshold(img_gray,217,255,cv2.THRESH_BINARY_INV)
#                                         # 경계값  218
# cv2.imshow('gray',img_gray)
# cv2.imshow('src',img_src)
# cv2.waitKey()
# cv2.destroyAllWindows()
#
# # ===================트랙바로 쓰레쏠드 써보기=====================
# import cv2
# def on_trackbar(x):
#     pass
#
# cv2.namedWindow('Threshold')
# cv2.createTrackbar('threshold', 'Threshold', 0, 255, on_trackbar)
# cv2.setTrackbarPos('threshold', 'Threshold', 128)
# img_src = cv2.imread('Myshape.png',cv2.IMREAD_COLOR)
# img_gray = cv2.cvtColor(img_src,cv2.COLOR_BGR2GRAY)
#
# while (1):
#     thres = cv2.getTrackbarPos('threshold', 'Threshold')
#     ret, img_binary = cv2.threshold(img_gray, thres, 255, cv2.THRESH_BINARY_INV)
#     cv2.imshow('Threshold', img_binary)
#
#     if cv2.waitKey(1) & 0xFF == 27: # ESC키
#         break
# cv2.destroyAllWindows()


# =============================외곽선(contours) ====================
# ====== 이미지 모멘트(질량중심) ========================================================
# import cv2
# img_src = cv2.imread('Myshape.png',cv2.IMREAD_COLOR)
# h,w=img_src.shape[:2]
# img_gray = cv2.cvtColor(img_src,cv2.COLOR_BGR2GRAY)
# img_gray=cv2.GaussianBlur(img_gray,(5,5),0)
# ret, img_binary = cv2.threshold(img_gray,230,255,cv2.THRESH_BINARY_INV)
# contours,hierarchy = cv2.findContours(img_binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
# # ===최대면적 area찾는 함수를위해 선언=========
# max_area = 0  # 최대 면적 초기화
# max_area_contour = None  # 최대 면적을 갖는 contour 초기화
#
# for i,contour in enumerate(contours):
#
#     area = cv2.contourArea(contour)
#     print( f"{i} = {area}")
#     if area>0 :
#
#         mu = cv2.moments(contour)
#         cx = int(mu['m10'] / (mu['m00']+1e-5))
#         cy = int(mu['m01'] / (mu['m00'] + 1e-5))
#
#         x, y, w, h = cv2.boundingRect(contour)  # 주어진 윤곽선을 감싸는 사각형을 찾음.
#
#         # cv2.rectangle(img_src, (x, y, w, h), (0, 0, 255), 2)
#         cv2.drawContours(img_src,[contour], 0, (0,255,0),2)
#         # cv2.putText(img_src,f"{(i)} : +{int(area)}",tuple(contour[0][0]),cv2.FONT_HERSHEY_PLAIN,1.0,(0,0,255),2)
#         # # 최소 면적의 사각형
#         # rect = cv2.minAreaRect(contour)
#         # box = cv2.boxPoints(rect)
#         # box = np.intp(box)
#         # cv2.drawContours(img_src, [box], 0, (255, 0, 0), 2)
#         #### 최소면적 사각형 그리기 끝
#         # 모멘트(중심)에 텍스트 표시
#         cv2.circle((img_src),(cx,cy),10,(255,255,255),-1)
#         cv2.putText(img_src, f'{i} = {area:.0f}', (cx - 40, cy + 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
#         # 그림 시작점에서 텍스트출력         cv2.putText(img_src, f'{i}:{area:.0f}', tuple(contour[0][0]),cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
#
#         # 최대 면적 갱신
#         if area > max_area:
#             max_area = area
#             max_area_contour = contour
#             MU = cv2.moments(max_area_contour)
#             CX = int(MU['m10'] / (MU['m00'] + 1e-5))
#             CY = int(MU['m01'] / (MU['m00'] + 1e-5))
#
#         cv2.imshow('src',img_src)
#
#         cv2.waitKey(0)
#         if cv2.waitKey(1) & 0xFF == 27: # ESC키
#             break
# if max_area_contour is not None:
#     cv2.drawContours(img_src, [max_area_contour], 0, (0, 0, 255), 3)
#     cv2.circle((img_src), (CX, CY), 10, (255, 255, 255), -1)
#     cv2.putText(img_src, f' Max Area = {max_area:.0f}', (CX - 150, CY ), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
#
# cv2.imshow('src',img_src)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # ========================코인그림 쓰레쏠드 이후 모폴로지 연산 =================
#
# import cv2
# import numpy as np
#
# img_src = cv2.imread('coin.png',cv2.IMREAD_COLOR)
# img_gray = cv2.cvtColor(img_src,cv2.COLOR_BGR2GRAY)
# img_gray = cv2.GaussianBlur(img_gray,(21,21),0)
#
# img_binary = cv2.adaptiveThreshold(img_gray,255,
#                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                cv2.THRESH_BINARY_INV,11,1)
#
#
# # ========================모폴로지 연산 =================
# # 확장하고 축소 : 클로징 / 축소하고 확장 :오프닝   *확장=흰색넓히기, 축소 =검정색넓히기
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
# img_dst = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN,kernel,iterations=4)
# img_dst2 = cv2.morphologyEx(img_dst, cv2.MORPH_CLOSE,kernel,iterations=5)
#
# # 계층 만들어서 안에 남은 점들 없애기 =================================
# contours,hierarchy = cv2.findContours(img_binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
#
#
#
# cv2.imshow('bi',img_binary)
# cv2.imshow('dst',img_dst)
# cv2.imshow('dst2',img_dst2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# # # =================컨터(외곽선)채우기 ===============================================================
# import cv2
#
# img_src = cv2.imread('coin.png',cv2.IMREAD_COLOR)
# height,width = img_src.shape[:2]
# # color to gray
# img_gray = cv2.cvtColor(img_src,cv2.COLOR_BGR2GRAY)
# # smoothing(bluring)
# img_gray = cv2.GaussianBlur(img_gray,(15,15), 0)
# img_binary = cv2.adaptiveThreshold(img_gray,255,
#                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                    cv2.THRESH_BINARY_INV, 11, 1)
# ################# 모폴로지 ####################
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
# img_binary = cv2.morphologyEx(img_binary,cv2.MORPH_OPEN,kernel,iterations=2)
# img_binary = cv2.morphologyEx(img_binary,cv2.MORPH_CLOSE,kernel,iterations=5)
# #############################################
# contours, hierarchy = cv2.findContours(img_binary,
#                                        cv2.RETR_CCOMP, # 계층 존재 2단계 아래 삭제
#                                        cv2.CHAIN_APPROX_NONE)
# for i,contour in enumerate(contours):
#     print(i, hierarchy[0][i])
#     # print(i, hierarchy[0][i][2])
#     if hierarchy[0][i][2] != -1:
#         cv2.drawContours(img_src,[contour],0,(0,255,0),2)
#         cv2.putText(img_src,str(i),tuple(contour[0][0]),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),1)
#
# cv2.imshow('src',img_src)
# cv2.imshow('dst',img_binary)
# # ======================오픈씨브이 보충수업 사진저장(롸이트 함수)
# cv2.imwrite('result.png',img_binary)
#
# cv2.waitKey()
# cv2.destroyAllWindows()


# ===================== 소벨 마스크 함수사용하기 (엣지 검출)

# import cv2
#
# scale = 1
# delta = 0
# ddepth = cv2.CV_16S
#
# # Load the image
# src_name = 'bicycle.jpg'
# src = cv2.imread(src_name, cv2.IMREAD_COLOR)
# # Check if image is loaded fine
# if src is None:
#     print('Error opening image: ' + src_name)
#     exit()
#
# src = cv2.GaussianBlur(src, (3, 3), 0)
# gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
# grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
# # Gradient-Y
# # grad_y = cv.Scharr(gray,ddepth,0,1)
# grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
#
# abs_grad_x = cv2.convertScaleAbs(grad_x)
# abs_grad_y = cv2.convertScaleAbs(grad_y)
#
# grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
#
#
# cv2.imshow('sobel-x', abs_grad_x)
# cv2.imshow('sobel-y', abs_grad_y)
# cv2.imshow('sobel-mixed', grad)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# ================ 라플라시안 검출
# ### 라플라시안
# import cv2
# ddepth = cv2.CV_16S
# kernel_size = 3
# # Load the image
# src_name = 'bicycle.jpg'
# img_src = cv2.imread(src_name, cv2.IMREAD_COLOR)
# # Check if image is loaded fine
# if img_src is None:
#     print('Error opening image: ' + src_name)
#     exit()
#
# img_src = cv2.GaussianBlur(img_src, (3, 3), 0)
# src_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
# dst = cv2.Laplacian(src_gray, ddepth, ksize=kernel_size)
# abs_dst = cv2.convertScaleAbs(dst)
# cv2.imshow('Laplasian', abs_dst)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ===========케니 엣지=========
### Canny Edge - with Trackbar
# import cv2
# import argparse
# max_lowThreshold = 100
# window_name = 'Edge Map'
# title_trackbar = 'Min Threshold:'
# ratio = 3
# kernel_size = 3
# def CannyThreshold(val):
#     low_threshold = val
#     img_blur = cv2.blur(src_gray, (3,3))
#     detected_edges = cv2.Canny(img_blur, low_threshold, low_threshold*ratio, kernel_size)
#     mask = detected_edges != 0
#     dst = img_src * (mask[:,:,None].astype(img_src.dtype))
#     cv2.imshow(window_name, dst)
#
# # Load the image
# src_name = 'bicycle.jpg'
# img_src = cv2.imread(src_name, cv2.IMREAD_COLOR)
# # Check if image is loaded fine
# if img_src is None:
#     print('Error opening image: ' + src_name)
#     exit()
# src_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
# cv2.namedWindow(window_name)
# cv2.createTrackbar(title_trackbar, window_name , 0, max_lowThreshold, CannyThreshold)
# CannyThreshold(15)
# cv2.waitKey()
# cv2.destroyAllWindows()


# ================================================엣지를  영상에서(중앙만) 검출
# import cv2
# import numpy as np
#
# scale = 1
# delta = 0
# ddepth = cv2.CV_16S
# capture=cv2.VideoCapture("lookbook.mp4")
# if capture.isOpened() == False:
#     print("동영상을 열 수 없음")
#     exit(1)
#
# while cv2.waitKey(10)<0:
#     ret,img_frame =capture.read()
#     h, w = img_frame.shape[:2]
#     if ret ==False:
#         print("캡쳐실패")
#         break
#
#     img_frame = cv2.GaussianBlur(img_frame, (3, 3), 0)
#     gray = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)
#     grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
#     # Gradient-Y
#     # grad_y = cv.Scharr(gray,ddepth,0,1)
#     grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
#
#     abs_grad_x = cv2.convertScaleAbs(grad_x)
#     abs_grad_y = cv2.convertScaleAbs(grad_y)
#
#     grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
#     h, w = grad.shape[:2]
#
#     grad2 = grad[h//4:3*h//4,w//4:3*w//4].copy()
#     grad2 = np.expand_dims(grad2, axis=-1)
#     img_frame[h//4:3*h//4,w//4:3*w//4,:] = grad2
#     cv2.imshow('res', img_frame)
#
#     key=cv2.waitKey(1)
#     if key==27:
#         break
# capture.release()
# cv2.destroyAllWindows()

#
# # =================================================hsv 트랙바======= 얼굴 영상
# import cv2
# import numpy as np
#
# def on_trackbar(x):
#     pass
#
# # trackbar에 대한 window를 별도로 생성
# cv2.namedWindow('Threshold Setting')
#
# # 해당 window(Threshold Setting)에 min_h에 대한 trackbar 생성
# cv2.createTrackbar('min_h', 'Threshold Setting', 0, 240, on_trackbar)
# cv2.setTrackbarPos('min_h', 'Threshold Setting', 10)
#
# # 해당 window(Threshold Setting)에 max_h에 대한 trackbar 생성
# cv2.createTrackbar('max_h', 'Threshold Setting', 0, 240, on_trackbar)
# cv2.setTrackbarPos('max_h', 'Threshold Setting', 50)
#
# # 해당 window(Threshold Setting)에 min_s에 대한 trackbar 생성
# cv2.createTrackbar('min_s', 'Threshold Setting', 0, 240, on_trackbar)
# cv2.setTrackbarPos('min_s', 'Threshold Setting', 50)
#
# # 해당 window(Threshold Setting)에 max_s에 대한 trackbar 생성
# cv2.createTrackbar('max_s', 'Threshold Setting', 0, 240, on_trackbar)
# cv2.setTrackbarPos('max_s', 'Threshold Setting', 50)
#
# # 해당 window(Threshold Setting)에 min_v에 대한 trackbar 생성
# cv2.createTrackbar('min_v', 'Threshold Setting', 0, 240, on_trackbar)
# cv2.setTrackbarPos('min_v', 'Threshold Setting', 50)
#
# # 해당 window(Threshold Setting)에 max_v에 대한 trackbar 생성
# cv2.createTrackbar('max_v', 'Threshold Setting', 0, 240, on_trackbar)
# cv2.setTrackbarPos('max_v', 'Threshold Setting', 50)
#
#
# capture = cv2.VideoCapture('lookbook.mp4')
# if capture.isOpened() == False:
#     print("동영상을 열 수 없음")
#     exit(1)
#
# key = -1
# while True:
#     ret,img_frame = capture.read()
#     print(f'1st. ret = {ret}, key = {key}')
#
#     # 영상 무한재생에 대한 코드
#     if not ret:
#         capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
#         continue
#
#     img_face = cv2.pyrDown(img_frame)
#
#     height, width = img_face.shape[:2]
#
#     img_face_display = img_face.copy()
#
#     img_hsv = cv2.cvtColor(img_face, cv2.COLOR_BGR2HSV)
#
#     # trackbar(Threshold Setting)에서 각 파라미터들에 대한 수치 조절을 실시간으로 행한다.
#     min_h = cv2.getTrackbarPos('min_h', 'Threshold Setting')
#     max_h = cv2.getTrackbarPos('max_h', 'Threshold Setting')
#     min_s = cv2.getTrackbarPos('min_s', 'Threshold Setting')
#     max_s = cv2.getTrackbarPos('max_s', 'Threshold Setting')
#     min_v = cv2.getTrackbarPos('min_v', 'Threshold Setting')
#     max_v = cv2.getTrackbarPos('max_v', 'Threshold Setting')
#
#     # 설정 오류에 대한 코드 (영상을 강제로 종료한다.)
#     if max_h < min_h:
#         print("설정 오류입니다.")
#         break
#
#     if max_s < min_s:
#         print("설정 오류입니다.")
#         break
#
#     if max_v < min_v:
#         print("설정 오류입니다.")
#         break
#
#     img_h, img_s, img_v = cv2.split(img_face)
#     face_mask = cv2.inRange(img_hsv, (min_h*(180/240), min_s*(255/240), min_v*(255/240)),
#                                      (max_h*(180/240), max_s*(255/240), max_v*(255/240)))
#
#     img_face = cv2.bitwise_and(img_hsv, img_hsv, mask=face_mask)
#     img_face = cv2.cvtColor(img_face, cv2.COLOR_HSV2BGR)
#     img_face_gray = cv2.cvtColor(img_face, cv2.COLOR_BGR2GRAY)
#
#     contours, hierarchy = cv2.findContours(img_face_gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
#     areas = []
#     for i, contour in enumerate(contours):
#         area = cv2.contourArea(contour)
#         areas.append(area)
#
#     for i, contour in enumerate(contours):
#         area = cv2.contourArea(contour)
#         if area == max(areas):
#             mu = cv2.moments(contour)
#             cx = int(mu['m10'] / (mu['m00'] + 1e-5))
#             cy = int(mu['m01'] / (mu['m00'] + 1e-5))
#             cv2.circle(img_face, (cx, cy), 3, (0, 0, 0), -1)
#
#             # 사각형 그리기
#             cv2.drawContours(img_face, [contour], 0, (0, 0, 0), 2)
#             x, y, w, h = cv2.boundingRect(contour)
#             cv2.rectangle(img_face_display, (x, y, w, h), (0, 255, 255), 2)
#             # 사각형 그리기 끝
#
#             # 최소 면적의 사각형 그리기
#             # rect = cv2.minAreaRect(contour)
#             # box = cv2.boxPoints(rect)
#             # box = np.int0(box)
#             # cv2.drawContours(img_face_display, [box], 0, (0, 255, 0), 2)
#             # 최소 면적의 사각형 그리기 끝
#
#             cv2.putText(img_face_display, f'{i} : {area:.2f}', (cx - 50, cy + 25),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#
#     if ret == False:
#         print("Video End")
#         break
#
#     cv2.imshow('Threshold', img_face_display)
#
#     key = cv2.waitKey(33)
#     print(f'2nd. ret = {ret}, key = {key}')
#
#     if key == 27:
#         break
#
# capture.release()
# cv2.destroyAllWindows()


# =================================================hsv 사진추출 연습

# =============HSV 전부 값 설정하기===================내 하던거
# import cv2
# img_src = cv2.imread('apple.jpg',cv2.IMREAD_COLOR)
# height, width = img_src.shape[:2]
#
# img_hsv = cv2.cvtColor(img_src,cv2.COLOR_BGR2HSV)
# h_min = 8; h_max = 20
# s_min = 0; s_max = 255
# v_min = 0; v_max = 255
# orange_mask = cv2.inRange(img_hsv, (h_min*(180/240), s_min*(255/240), v_min*(255/240)),
#                                       (h_max*(180/240), s_max*(255/240), v_max*(255/240)))
# img_orange = cv2.bitwise_and(img_hsv,img_hsv,mask=orange_mask)
# img_orange = cv2.cvtColor(img_orange,cv2.COLOR_HSV2BGR)
#
#
# cv2.imshow('src',img_orange)
# cv2.imshow('h',orange_mask)
#
# cv2.waitKey()
# cv2.destroyAllWindows()
#

# # ==================================(사과 hsv 트랙바)
# # 트랙바 콜백 함수 생성
# ### RED 추출
# import cv2
#
# def on_trackbar(x):
#     pass
#
# # trackbar에 대한 window를 별도로 생성
# cv2.namedWindow('Threshold Setting')
#
# # 해당 window(Threshold Setting)에 s_min에 대한 trackbar 생성
# cv2.createTrackbar('s_min', 'Threshold Setting', 0, 255, on_trackbar)
# cv2.setTrackbarPos('s_min', 'Threshold Setting', 50)
#
# # 해당 window(Threshold Setting)에 s_max에 대한 trackbar 생성
# cv2.createTrackbar('s_max', 'Threshold Setting', 0, 255, on_trackbar)
# cv2.setTrackbarPos('s_max', 'Threshold Setting', 50)
#
# # 해당 window(Threshold Setting)에 v_min에 대한 trackbar 생성
# cv2.createTrackbar('v_min', 'Threshold Setting', 0, 255, on_trackbar)
# cv2.setTrackbarPos('v_min', 'Threshold Setting', 50)
#
# # 해당 window(Threshold Setting)에 v_max에 대한 trackbar 생성
# cv2.createTrackbar('v_max', 'Threshold Setting', 0, 255, on_trackbar)
# cv2.setTrackbarPos('v_max', 'Threshold Setting', 50)
#
# while True:
#     img_src = cv2.imread('apple.jpg', cv2.IMREAD_COLOR)
#     height, width = img_src.shape[:2]
#
#     # hsv 이미지를 매번 새로 업로드 시켜준다.
#     # 그렇게하지 않으면 s_min, s_max, v_min, v_max에 따라 기존 이미지에 갱신되기 때문!
#     img_hsv = cv2.cvtColor(img_src, cv2.COLOR_BGR2HSV)
#
#     h_min1 = 0
#     h_max1 = 5
#     h_min2 = 170
#     h_max2 = 180
#
#     s_min = cv2.getTrackbarPos('s_min', 'Threshold Setting')
#     s_max = cv2.getTrackbarPos('s_max', 'Threshold Setting')
#     v_min = cv2.getTrackbarPos('v_min', 'Threshold Setting')
#     v_max = cv2.getTrackbarPos('v_max', 'Threshold Setting')
#
#     # 설정 오류 검출을 위한 반복문 탈출
#     if s_max < s_min:
#         print("설정 오류입니다.")
#         break
#
#     if v_max < v_min:
#         print("설정 오류입니다.")
#         break
#
#     red_mask_low = cv2.inRange(img_hsv, (h_min1, s_min, v_min), (h_max1, s_max, v_max))
#     red_mask_high = cv2.inRange(img_hsv, (h_min2, s_min, v_min), (h_max2, s_max, v_max))
#
#     red_mask = cv2.addWeighted(red_mask_low, 1.0, red_mask_high, 1.0, 0.0)
#     img_red = cv2.bitwise_and(img_hsv, img_hsv, mask=red_mask)
#     img_red = cv2.cvtColor(img_red, cv2.COLOR_HSV2BGR)
#
#     cv2.imshow('src', img_red)
#     cv2.imshow('h', red_mask)
#
#     img_red_gray = cv2.cvtColor(img_red, cv2.COLOR_BGR2GRAY)
#     contours, hierarchy = cv2.findContours(img_red_gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
#
#     for i, contour in enumerate(contours):
#         area = cv2.contourArea(contour)
#         if area > 10000:
#             mu = cv2.moments(contour)
#             cx = int(mu['m10'] / (mu['m00'] + 1e-5))
#             cy = int(mu['m01'] / (mu['m00'] + 1e-5))
#             cv2.circle(img_red, (cx, cy), 3, (0, 0, 0), -1)
#
#             # 사각형 그리기
#             # cv2.drawContours(img_src, [contour], 0, (0, 0, 0), 2)
#             x, y, w, h = cv2.boundingRect(contour)
#             cv2.rectangle(img_src, (x, y, w, h), (0, 255, 255), 2)
#             # 사각형 그리기 끝
#
#             # 최소 면적의 사각형 그리기
#             # rect = cv2.minAreaRect(contour)
#             # box = cv2.boxPoints(rect)
#             # box = np.int0(box)
#             # cv2.drawContours(img_apple_display, [box], 0, (0, 255, 0), 2)
#             # 최소 면적의 사각형 그리기 끝
#
#             cv2.circle(img_src,(cx - 50, cy + 25),3,(0,0,255),-1)
#             cv2.putText(img_src, f'{i} : {area:.2f}', (cx - 50, cy + 25),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#
#             cv2.imshow("final result",img_src)
#
#     # 키보드 입력 대기 시간 : 1ms / ESC(27) 누를시 종료
#     key = cv2.waitKey(1)
#     if key == 27:
#         break
#
# cv2.destroyAllWindows()


# #  코인 타원으로 감싸기 ======================================
# img_src = cv2.imread('coin.png',cv2.IMREAD_COLOR)
# height,width = img_src.shape[:2]
# # color to gray
# img_gray = cv2.cvtColor(img_src,cv2.COLOR_BGR2GRAY)
# # smoothing(bluring)
# img_gray = cv2.GaussianBlur(img_gray,(15,15), 0)
# img_binary = cv2.adaptiveThreshold(img_gray,255,
#                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                    cv2.THRESH_BINARY_INV, 11, 1)
# ################# 모폴로지 ####################
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
# img_binary = cv2.morphologyEx(img_binary,cv2.MORPH_OPEN,kernel,iterations=2)
# img_binary = cv2.morphologyEx(img_binary,cv2.MORPH_CLOSE,kernel,iterations=5)
# #############################################
# contours, hierarchy = cv2.findContours(img_binary,
#                                        cv2.RETR_CCOMP, # 계층 존재 2단계 아래 삭제
#                                        cv2.CHAIN_APPROX_NONE)
# for i,contour in enumerate(contours):
#     print(i, hierarchy[0][i])
#     # print(i, hierarchy[0][i][2])
#     if hierarchy[0][i][2] != -1:
#         # cv2.drawContours(img_src,[contour],0,(0,255,0),2)
#         cv2.putText(img_src,str(i),tuple(contour[0][0]),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),1)
#         ellipse = cv2.fitEllipse(contour)
#         cv2.ellipse(img_src,ellipse,(0,255,0), 2)
#
# cv2.imshow('src',img_src)
# cv2.imshow('dst',img_binary)
#
# cv2.waitKey()
# cv2.destroyAllWindows()



# 231129 시험 1번 =====사진 줄이고 이동하고 돌리기 ==================
# import cv2
# import numpy as np
#
# img = cv2.imread('love.png', cv2.IMREAD_COLOR)
# height, width = img.shape[:2]
#
# scale_factor = 0.707
# scaling_matrix = np.array([[scale_factor, 0, 0], [0, scale_factor, 0], [0, 0, 1]])
# translation_matrix = np.array([[1, 0, width*(1-scale_factor)/2], [0, 1,height*(1-scale_factor)/2], [0, 0, 1]])
# angle = 45
# radian = angle * np.pi / 180
# c = np.cos(radian)
# s = np.sin(radian)
# center_x = width / 2
# center_y = height / 2
# rotation_matrix = np.array(
#     [[c, s, (1 - c) * center_x - s * center_y], [-s, c, s * center_x + (1 - c) * center_y], [0, 0, 1]])
#
# # 정해진 순서대로 변환 행렬을 곱하여 하나의 행렬을 생성합니다.
# T = np.eye(3)
# T = np.dot(scaling_matrix, T)
# T = np.dot(translation_matrix, T)
# T = np.dot(rotation_matrix, T)
#
# dst = np.zeros((height, width, img.shape[2]), dtype=np.uint8)
#
# for y in range(height):
#     for x in range(width):
#
#         # 미리 구해놓은 변환행렬을 행렬곱 한번으로 적용합니다.
#         # 여기에서도 backward mapping을 사용합니다.
#         new_p = np.array([x, y, 1])
#         inv_scaling_matrix = np.linalg.inv(T)
#         old_p = np.dot(inv_scaling_matrix, new_p)
#
#         x_, y_ = old_p[:2]
#         x_ = int(x_)
#         y_ = int(y_)
#
#         if x_ > 0 and x_ < width and y_ > 0 and y_ < height:
#             dst[y, x] = img[y_, x_]
#
# cv2.imshow("result", dst)
# cv2.waitKey(0)

# 231129 시험 2번 =====외관선 추출해서  컨터 추출,면적제일큰 컨터에리어 표시 모멘트원으로표시 모멘트아래에 면적 숫자로표시(text) ==================

# import cv2
# img_src = cv2.imread('Myshape.png',cv2.IMREAD_COLOR)
# h,w=img_src.shape[:2]
# img_gray = cv2.cvtColor(img_src,cv2.COLOR_BGR2GRAY)
# img_gray=cv2.GaussianBlur(img_gray,(5,5),0)
# ret, img_binary = cv2.threshold(img_gray,230,255,cv2.THRESH_BINARY_INV)
# contours,hierarchy = cv2.findContours(img_binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
# # ===최대면적 area찾는 함수를위해 선언=========
# max_area = 0  # 최대 면적 초기화
# max_area_contour = None  # 최대 면적을 갖는 contour 초기화
#
# for i,contour in enumerate(contours):
#
#     area = cv2.contourArea(contour)
#     print( f"{i} = {area}")
#     if area>0 :
#
#         mu = cv2.moments(contour)
#         cx = int(mu['m10'] / (mu['m00']+1e-5))
#         cy = int(mu['m01'] / (mu['m00'] + 1e-5))
#
#         x, y, w, h = cv2.boundingRect(contour)  # 주어진 윤곽선을 감싸는 사각형을 찾음.
#
#         # cv2.rectangle(img_src, (x, y, w, h), (0, 0, 255), 2)
#         cv2.drawContours(img_src,[contour], 0, (0,255,0),2)
#         # cv2.putText(img_src,f"{(i)} : +{int(area)}",tuple(contour[0][0]),cv2.FONT_HERSHEY_PLAIN,1.0,(0,0,255),2)
#         # # 최소 면적의 사각형
#         # rect = cv2.minAreaRect(contour)
#         # box = cv2.boxPoints(rect)
#         # box = np.intp(box)
#         # cv2.drawContours(img_src, [box], 0, (255, 0, 0), 2)
#         #### 최소면적 사각형 그리기 끝
#         # 모멘트(중심)에 텍스트 표시
#         cv2.circle((img_src),(cx,cy),4,(255,255,255),-1)
#         cv2.putText(img_src, f'{i} = {area:.0f}', (cx - 40, cy + 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
#         # 그림 시작점에서 텍스트출력         cv2.putText(img_src, f'{i}:{area:.0f}', tuple(contour[0][0]),cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
#
#         # 최대 면적 갱신
#         if area > max_area:
#             max_area = area
#             max_area_contour = contour
#             MU = cv2.moments(max_area_contour)
#             CX = int(MU['m10'] / (MU['m00'] + 1e-5))
#             CY = int(MU['m01'] / (MU['m00'] + 1e-5))
#
#         cv2.imshow('src',img_src)
#
#         cv2.waitKey(0)
#         if cv2.waitKey(1) & 0xFF == 27: # ESC키
#             break
# if max_area_contour is not None:
#     cv2.drawContours(img_src, [max_area_contour], 0, (0, 0, 255), 3)
#     cv2.circle((img_src), (CX, CY), 10, (0, 0, 255), -1)
#     cv2.putText(img_src, f' Max Area = {max_area:.0f}', (CX - 150, CY+30 ), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
#
# cv2.imshow('src',img_src)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




#231129 시험 3번 ==================(사과 사각형으로 추출완료. hsv 트랙바)
# 트랙바 콜백 함수 생성
# ## RED 추출
# import cv2
#
# def on_trackbar(x):
#     pass
#
# # trackbar에 대한 window를 별도로 생성
# cv2.namedWindow('Threshold Setting')
#
# # 해당 window(Threshold Setting)에 s_min에 대한 trackbar 생성
# cv2.createTrackbar('s_min', 'Threshold Setting', 0, 255, on_trackbar)
# cv2.setTrackbarPos('s_min', 'Threshold Setting', 50)
#
# # 해당 window(Threshold Setting)에 s_max에 대한 trackbar 생성
# cv2.createTrackbar('s_max', 'Threshold Setting', 0, 255, on_trackbar)
# cv2.setTrackbarPos('s_max', 'Threshold Setting', 50)
#
# # 해당 window(Threshold Setting)에 v_min에 대한 trackbar 생성
# cv2.createTrackbar('v_min', 'Threshold Setting', 0, 255, on_trackbar)
# cv2.setTrackbarPos('v_min', 'Threshold Setting', 50)
#
# # 해당 window(Threshold Setting)에 v_max에 대한 trackbar 생성
# cv2.createTrackbar('v_max', 'Threshold Setting', 0, 255, on_trackbar)
# cv2.setTrackbarPos('v_max', 'Threshold Setting', 50)
#
# while True:
#     img_src = cv2.imread('apple.jpg', cv2.IMREAD_COLOR)
#     height, width = img_src.shape[:2]
#
#     # hsv 이미지를 매번 새로 업로드 시켜준다.
#     # 그렇게하지 않으면 s_min, s_max, v_min, v_max에 따라 기존 이미지에 갱신되기 때문!
#     img_hsv = cv2.cvtColor(img_src, cv2.COLOR_BGR2HSV)
#
#     h_min1 = 0
#     h_max1 = 5
#     h_min2 = 170
#     h_max2 = 180
#
#     s_min = cv2.getTrackbarPos('s_min', 'Threshold Setting')
#     s_max = cv2.getTrackbarPos('s_max', 'Threshold Setting')
#     v_min = cv2.getTrackbarPos('v_min', 'Threshold Setting')
#     v_max = cv2.getTrackbarPos('v_max', 'Threshold Setting')
#
#     # 설정 오류 검출을 위한 반복문 탈출
#     if s_max < s_min:
#         print("설정 오류입니다.")
#         break
#
#     if v_max < v_min:
#         print("설정 오류입니다.")
#         break
#
#     red_mask_low = cv2.inRange(img_hsv, (h_min1, s_min, v_min), (h_max1, s_max, v_max))
#     red_mask_high = cv2.inRange(img_hsv, (h_min2, s_min, v_min), (h_max2, s_max, v_max))
#
#     red_mask = cv2.addWeighted(red_mask_low, 1.0, red_mask_high, 1.0, 0.0)
#     img_red = cv2.bitwise_and(img_hsv, img_hsv, mask=red_mask)
#     img_red = cv2.cvtColor(img_red, cv2.COLOR_HSV2BGR)
#
#     cv2.imshow('src', img_red)
#     cv2.imshow('h', red_mask)
#
#     img_red_gray = cv2.cvtColor(img_red, cv2.COLOR_BGR2GRAY)
#     contours, hierarchy = cv2.findContours(img_red_gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
#
#     for i, contour in enumerate(contours):
#         area = cv2.contourArea(contour)
#         if area > 10000:
#             mu = cv2.moments(contour)
#             cx = int(mu['m10'] / (mu['m00'] + 1e-5))
#             cy = int(mu['m01'] / (mu['m00'] + 1e-5))
#             cv2.circle(img_red, (cx, cy), 3, (0, 0, 0), -1)
#
#             # 사각형 그리기
#             # cv2.drawContours(img_src, [contour], 0, (0, 0, 0), 2)
#             x, y, w, h = cv2.boundingRect(contour)
#             cv2.rectangle(img_src, (x, y, w, h), (0, 255, 255), 2)
#             # 사각형 그리기 끝
#
#             # 최소 면적의 사각형 그리기
#             # rect = cv2.minAreaRect(contour)
#             # box = cv2.boxPoints(rect)
#             # box = np.int0(box)
#             # cv2.drawContours(img_apple_display, [box], 0, (0, 255, 0), 2)
#             # 최소 면적의 사각형 그리기 끝
#
#             cv2.circle(img_src,(cx - 50, cy + 25),3,(0,0,255),-1)
#             cv2.putText(img_src, f'{i} : {area:.2f}', (cx - 50, cy + 25),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#
#             cv2.imshow("final result",img_src)
#
#     # 키보드 입력 대기 시간 : 1ms / ESC(27) 누를시 종료
#     key = cv2.waitKey(1)
#     if key == 27:
#         break
#
# cv2.destroyAllWindows()

# 231129 진짜시험==================================================================
# =================================================================================
# 1번문제

# # # 45도 사진회전하기
# import cv2
# import numpy as np
#
# img = cv2.imread('love.png', cv2.IMREAD_COLOR)
# height, width = img.shape[:2]
#
# # 45도를 라디안으로 변환하여 코싸인값과 싸인값을 구합니다.
# angle = 0   # 회전없으면 0도
# radian = angle * np.pi / 180
# c = np.cos(radian)
# s = np.sin(radian)
#
# # 회전변환행렬을 구성합니다.
# # OpenCV의 원점이 왼쪽아래가 아니라 왼쪽위라서 [[c, -s, 0], [s, c, 0]]가 아니라
# # [[c, s, 0], [-s, c, 0]]입니다.
# rotation_matrix = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=float)
#
# dst = np.zeros(img.shape, dtype=np.uint8)
#
# for y in range(height - 1):
#     for x in range(width - 1):
#
#         # backward mapping
#         # 결과 이미지의 픽셀 new_p로 이동하는 입력 이미지의
#         # 픽셀 old_p의 위치를 계산합니다.
#         new_p = np.array([x-150, y-50, 1])  # 오른쪽 150.  아래 50 이동
#         inv_rotation_matrix = np.linalg.inv(rotation_matrix)
#         old_p = np.dot(inv_rotation_matrix, new_p)
#
#         # new_p 위치에 계산하여 얻은 old_p 픽셀의 값을 대입합니다.
#         x_, y_ = old_p[:2]
#         x_ = int(x_)
#         y_ = int(y_)
#
#         # 입력 이미지의 픽셀을 가져올 수 있는 경우에만
#         # 결과 이미지의 현재 위치의 픽셀로 사용합니다.
#         if x_ > 0 and x_ < width and y_ > 0 and y_ < height:
#             dst[y, x] = img[y_, x_]
#
# result = cv2.hconcat([img, dst])
# cv2.imshow("result", result)
# cv2.waitKey(0)

# 2번===============================================================
#
# import cv2
# import numpy as np
#
# img_src = cv2.imread('love.png', cv2.IMREAD_COLOR)
# img = cv2.pyrDown(img_src)
# height, width = img.shape[:2]
# # 0.7배율 형성해서 붙이기====
#
# scale_factor = 0.7
# scaling_matrix = np.array([[scale_factor, 0, 0], [0, scale_factor, 0], [0, 0, 1]])
# # 정해진 순서대로 변환 행렬을 곱하여 하나의 행렬을 생성합니다.
# T = np.eye(3)
# T = np.dot(scaling_matrix, T)
#
# dst = np.zeros((height, width, img.shape[2]), dtype=np.uint8)
#
# for y in range(height):
#     for x in range(width):
#
#         # 미리 구해놓은 변환행렬을 행렬곱 한번으로 적용합니다.
#         # 여기에서도 backward mapping을 사용합니다.
#         new_p = np.array([x, y, 1])
#         inv_scaling_matrix = np.linalg.inv(T)
#         old_p = np.dot(inv_scaling_matrix, new_p)
#
#         x_, y_ = old_p[:2]
#         x_ = int(x_)
#         y_ = int(y_)
#
#         if x_ > 0 and x_ < width and y_ > 0 and y_ < height:
#             dst[y, x] = img[y_, x_]
# result = cv2.hconcat([img, dst])
#
# # 0.3배율도 형성해서 붙이기====
# height, width = img.shape[:2]
# scale_factor = 0.3
# scaling_matrix = np.array([[scale_factor, 0, 0], [0, scale_factor, 0], [0, 0, 1]])
# # 정해진 순서대로 변환 행렬을 곱하여 하나의 행렬을 생성합니다.
# T = np.eye(3)
# T = np.dot(scaling_matrix, T)
#
# dst2 = np.zeros((height, width, img.shape[2]), dtype=np.uint8)
#
# for y in range(height):
#     for x in range(width):
#
#         # 미리 구해놓은 변환행렬을 행렬곱 한번으로 적용합니다.
#         # 여기에서도 backward mapping을 사용합니다.
#         new_p = np.array([x, y, 1])
#         inv_scaling_matrix = np.linalg.inv(T)
#         old_p = np.dot(inv_scaling_matrix, new_p)
#
#         x_, y_ = old_p[:2]
#         x_ = int(x_)
#         y_ = int(y_)
#
#         if x_ > 0 and x_ < width and y_ > 0 and y_ < height:
#             dst2[y, x] = img[y_, x_]
# result = cv2.hconcat([result, dst2])
#
# cv2.imshow("result", result)
# cv2.waitKey(0)


# 3번 ===================================

# import cv2
# import numpy as np
#
# img = cv2.imread('love.png', cv2.IMREAD_COLOR)
# img=cv2.pyrDown(img)    #사진 너무커서 사이즈 줄이고 시작.
# height, width = img.shape[:2]
# scale_factor = 1
# scaling_matrix = np.array([[scale_factor, 0, 0], [0, scale_factor, 0], [0, 0, 1]])
# # 정해진 순서대로 변환 행렬을 곱하여 하나의 행렬을 생성합니다.
# T = np.eye(3)
# T = np.dot(scaling_matrix, T)
# dst = np.zeros((height, width, img.shape[2]), dtype=np.uint8)
#
# for y in range(height):
#     for x in range(width):
#
#         # 미리 구해놓은 변환행렬을 행렬곱 한번으로 적용합니다.
#         # 여기에서도 backward mapping을 사용합니다.
#         new_p = np.array([x, y, 1])
#         inv_scaling_matrix = np.linalg.inv(T)
#         old_p = np.dot(inv_scaling_matrix, new_p)
#
#         x_, y_ = old_p[:2]
#         x_ = int(x_)
#         y_ = int(y_)
#
#         if x_ > 0 and x_ < width and y_ > 0 and y_ < height:
#             dst[y, x] = img[y_, x_]
# # 60도 붙이기
# scale_factor =1
# scaling_matrix = np.array([[scale_factor, 0, 0], [0, scale_factor, 0], [0, 0, 1]])
# translation_matrix = np.array([[1, 0, width*(1-scale_factor)/2], [0, 1,height*(1-scale_factor)/2], [0, 0, 1]])
# angle = 60
# radian = angle * np.pi / 180
# c = np.cos(radian)
# s = np.sin(radian)
# center_x = width / 2
# center_y = height / 2
# rotation_matrix = np.array(
#     [[c, s, (1 - c) * center_x - s * center_y], [-s, c, s * center_x + (1 - c) * center_y], [0, 0, 1]])
#
# # 정해진 순서대로 변환 행렬을 곱하여 하나의 행렬을 생성합니다.
# T = np.eye(3)
# T = np.dot(scaling_matrix, T)
# T = np.dot(translation_matrix, T)
# T = np.dot(rotation_matrix, T)
#
# dst2 = np.zeros((height, width, img.shape[2]), dtype=np.uint8)
#
# for y in range(height):
#     for x in range(width):
#
#         # 미리 구해놓은 변환행렬을 행렬곱 한번으로 적용합니다.
#         # 여기에서도 backward mapping을 사용합니다.
#         new_p = np.array([x, y, 1])
#         inv_scaling_matrix = np.linalg.inv(T)
#         old_p = np.dot(inv_scaling_matrix, new_p)
#
#         x_, y_ = old_p[:2]
#         x_ = int(x_)
#         y_ = int(y_)
#
#         if x_ > 0 and x_ < width and y_ > 0 and y_ < height:
#             dst2[y, x] = img[y_, x_]
#
# result = cv2.hconcat([dst,dst2])
#
#
# # 120도붙이기======
# scale_factor =1
# scaling_matrix = np.array([[scale_factor, 0, 0], [0, scale_factor, 0], [0, 0, 1]])
# translation_matrix = np.array([[1, 0, width*(1-scale_factor)/2], [0, 1,height*(1-scale_factor)/2], [0, 0, 1]])
# angle = 120
# radian = angle * np.pi / 180
# c = np.cos(radian)
# s = np.sin(radian)
# center_x = width / 2
# center_y = height / 2
# rotation_matrix = np.array(
#     [[c, s, (1 - c) * center_x - s * center_y], [-s, c, s * center_x + (1 - c) * center_y], [0, 0, 1]])
#
# # 정해진 순서대로 변환 행렬을 곱하여 하나의 행렬을 생성합니다.
# T = np.eye(3)
# T = np.dot(scaling_matrix, T)
# T = np.dot(translation_matrix, T)
# T = np.dot(rotation_matrix, T)
#
# dst2 = np.zeros((height, width, img.shape[2]), dtype=np.uint8)
#
# for y in range(height):
#     for x in range(width):
#
#         # 미리 구해놓은 변환행렬을 행렬곱 한번으로 적용합니다.
#         # 여기에서도 backward mapping을 사용합니다.
#         new_p = np.array([x, y, 1])
#         inv_scaling_matrix = np.linalg.inv(T)
#         old_p = np.dot(inv_scaling_matrix, new_p)
#
#         x_, y_ = old_p[:2]
#         x_ = int(x_)
#         y_ = int(y_)
#
#         if x_ > 0 and x_ < width and y_ > 0 and y_ < height:
#             dst2[y, x] = img[y_, x_]
#
# result = cv2.hconcat([result,dst2])
#
# # 180도================
# scale_factor =1
# scaling_matrix = np.array([[scale_factor, 0, 0], [0, scale_factor, 0], [0, 0, 1]])
# translation_matrix = np.array([[1, 0, width*(1-scale_factor)/2], [0, 1,height*(1-scale_factor)/2], [0, 0, 1]])
# angle = 180
# radian = angle * np.pi / 180
# c = np.cos(radian)
# s = np.sin(radian)
# center_x = width / 2
# center_y = height / 2
# rotation_matrix = np.array(
#     [[c, s, (1 - c) * center_x - s * center_y], [-s, c, s * center_x + (1 - c) * center_y], [0, 0, 1]])
#
# # 정해진 순서대로 변환 행렬을 곱하여 하나의 행렬을 생성합니다.
# T = np.eye(3)
# T = np.dot(scaling_matrix, T)
# T = np.dot(translation_matrix, T)
# T = np.dot(rotation_matrix, T)
#
# dst2 = np.zeros((height, width, img.shape[2]), dtype=np.uint8)
#
# for y in range(height):
#     for x in range(width):
#
#         # 미리 구해놓은 변환행렬을 행렬곱 한번으로 적용합니다.
#         # 여기에서도 backward mapping을 사용합니다.
#         new_p = np.array([x, y, 1])
#         inv_scaling_matrix = np.linalg.inv(T)
#         old_p = np.dot(inv_scaling_matrix, new_p)
#
#         x_, y_ = old_p[:2]
#         x_ = int(x_)
#         y_ = int(y_)
#
#         if x_ > 0 and x_ < width and y_ > 0 and y_ < height:
#             dst2[y, x] = img[y_, x_]
#
# result2 = dst2
#
# # 240도 ========
# scale_factor =1
# scaling_matrix = np.array([[scale_factor, 0, 0], [0, scale_factor, 0], [0, 0, 1]])
# translation_matrix = np.array([[1, 0, width*(1-scale_factor)/2], [0, 1,height*(1-scale_factor)/2], [0, 0, 1]])
# angle = 240
# radian = angle * np.pi / 180
# c = np.cos(radian)
# s = np.sin(radian)
# center_x = width / 2
# center_y = height / 2
# rotation_matrix = np.array(
#     [[c, s, (1 - c) * center_x - s * center_y], [-s, c, s * center_x + (1 - c) * center_y], [0, 0, 1]])
#
# # 정해진 순서대로 변환 행렬을 곱하여 하나의 행렬을 생성합니다.
# T = np.eye(3)
# T = np.dot(scaling_matrix, T)
# T = np.dot(translation_matrix, T)
# T = np.dot(rotation_matrix, T)
#
# dst2 = np.zeros((height, width, img.shape[2]), dtype=np.uint8)
#
# for y in range(height):
#     for x in range(width):
#
#         # 미리 구해놓은 변환행렬을 행렬곱 한번으로 적용합니다.
#         # 여기에서도 backward mapping을 사용합니다.
#         new_p = np.array([x, y, 1])
#         inv_scaling_matrix = np.linalg.inv(T)
#         old_p = np.dot(inv_scaling_matrix, new_p)
#
#         x_, y_ = old_p[:2]
#         x_ = int(x_)
#         y_ = int(y_)
#
#         if x_ > 0 and x_ < width and y_ > 0 and y_ < height:
#             dst2[y, x] = img[y_, x_]
#
# result2 = cv2.hconcat([result2,dst2])
# # 300도 ========
# scale_factor =1
# scaling_matrix = np.array([[scale_factor, 0, 0], [0, scale_factor, 0], [0, 0, 1]])
# translation_matrix = np.array([[1, 0, width*(1-scale_factor)/2], [0, 1,height*(1-scale_factor)/2], [0, 0, 1]])
# angle = 300
# radian = angle * np.pi / 180
# c = np.cos(radian)
# s = np.sin(radian)
# center_x = width / 2
# center_y = height / 2
# rotation_matrix = np.array(
#     [[c, s, (1 - c) * center_x - s * center_y], [-s, c, s * center_x + (1 - c) * center_y], [0, 0, 1]])
#
# # 정해진 순서대로 변환 행렬을 곱하여 하나의 행렬을 생성합니다.
# T = np.eye(3)
# T = np.dot(scaling_matrix, T)
# T = np.dot(translation_matrix, T)
# T = np.dot(rotation_matrix, T)
#
# dst2 = np.zeros((height, width, img.shape[2]), dtype=np.uint8)
#
# for y in range(height):
#     for x in range(width):
#
#         # 미리 구해놓은 변환행렬을 행렬곱 한번으로 적용합니다.
#         # 여기에서도 backward mapping을 사용합니다.
#         new_p = np.array([x, y, 1])
#         inv_scaling_matrix = np.linalg.inv(T)
#         old_p = np.dot(inv_scaling_matrix, new_p)
#
#         x_, y_ = old_p[:2]
#         x_ = int(x_)
#         y_ = int(y_)
#
#         if x_ > 0 and x_ < width and y_ > 0 and y_ < height:
#             dst2[y, x] = img[y_, x_]
#
# result2 = cv2.hconcat([result2,dst2])
# result = cv2.vconcat([result,result2])
#
# cv2.imshow("result", result)
# cv2.waitKey(0)

# 4번 =================
# import cv2
# import numpy as np
#
# img = cv2.imread('love.png', cv2.IMREAD_COLOR)
# img=cv2.pyrDown(img)    #사진 너무커서 사이즈 줄이고 시작.
# height, width = img.shape[:2]
#
# # 원본과 0.7배 합치기
# scale_factor = 0.7
# scaling_matrix = np.array([[scale_factor, 0, 0], [0, scale_factor, 0], [0, 0, 1]])
# translation_matrix = np.array([[1, 0, 0], [0, 1,0], [0, 0, 1]])
# angle = 0
# radian = angle * np.pi / 180
# c = np.cos(radian)
# s = np.sin(radian)
# center_x = width / 2
# center_y = height / 2
# rotation_matrix = np.array(
#     [[c, s, (1 - c) * center_x - s * center_y], [-s, c, s * center_x + (1 - c) * center_y], [0, 0, 1]])
#
# # 정해진 순서대로 변환 행렬을 곱하여 하나의 행렬을 생성합니다.
# T = np.eye(3)
# T = np.dot(scaling_matrix, T)
# T = np.dot(translation_matrix, T)
# T = np.dot(rotation_matrix, T)
#
# dst = np.zeros((height, width, img.shape[2]), dtype=np.uint8)
#
# for y in range(height):
#     for x in range(width):
#
#         # 미리 구해놓은 변환행렬을 행렬곱 한번으로 적용합니다.
#         # 여기에서도 backward mapping을 사용합니다.
#         new_p = np.array([x, y, 1])
#         inv_scaling_matrix = np.linalg.inv(T)
#         old_p = np.dot(inv_scaling_matrix, new_p)
#
#         x_, y_ = old_p[:2]
#         x_ = int(x_)
#         y_ = int(y_)
#
#         if x_ > 0 and x_ < width and y_ > 0 and y_ < height:
#             dst[y, x] = img[y_, x_]
# result = cv2.hconcat([img,dst])  # 1번,2번 가로로 합성완료
#
#
# # 15%이동
# scale_factor = 0.7
# scaling_matrix = np.array([[scale_factor, 0, 0], [0, scale_factor, 0], [0, 0, 1]])
# translation_matrix = np.array([[1, 0, width*0.15], [0, 1,height*0.15], [0, 0, 1]])
# angle = 0
# radian = angle * np.pi / 180
# c = np.cos(radian)
# s = np.sin(radian)
# center_x = width / 2
# center_y = height / 2
# rotation_matrix = np.array(
#     [[c, s, (1 - c) * center_x - s * center_y], [-s, c, s * center_x + (1 - c) * center_y], [0, 0, 1]])
#
# # 정해진 순서대로 변환 행렬을 곱하여 하나의 행렬을 생성합니다.
# T = np.eye(3)
# T = np.dot(scaling_matrix, T)
# T = np.dot(translation_matrix, T)
# T = np.dot(rotation_matrix, T)
#
# dst2 = np.zeros((height, width, img.shape[2]), dtype=np.uint8)
#
# for y in range(height):
#     for x in range(width):
#
#         # 미리 구해놓은 변환행렬을 행렬곱 한번으로 적용합니다.
#         # 여기에서도 backward mapping을 사용합니다.
#         new_p = np.array([x, y, 1])
#         inv_scaling_matrix = np.linalg.inv(T)
#         old_p = np.dot(inv_scaling_matrix, new_p)
#
#         x_, y_ = old_p[:2]
#         x_ = int(x_)
#         y_ = int(y_)
#
#         if x_ > 0 and x_ < width and y_ > 0 and y_ < height:
#             dst2[y, x] = img[y_, x_]
# result2 = dst2
#
# # 45도 회전 한걸  "15%이동한사진" 과 붙이기
# scale_factor = 0.7
# scaling_matrix = np.array([[scale_factor, 0, 0], [0, scale_factor, 0], [0, 0, 1]])
# translation_matrix = np.array([[1, 0, width*(1-scale_factor)/2], [0, 1,height*(1-scale_factor)/2], [0, 0, 1]])
#                                                                     # 회전중심 중앙 맞추기
# angle = 45
# radian = angle * np.pi / 180
# c = np.cos(radian)
# s = np.sin(radian)
# center_x = width / 2
# center_y = height / 2
# rotation_matrix = np.array(
#     [[c, s, (1 - c) * center_x - s * center_y], [-s, c, s * center_x + (1 - c) * center_y], [0, 0, 1]])
#
# # 정해진 순서대로 변환 행렬을 곱하여 하나의 행렬을 생성합니다.
# T = np.eye(3)
# T = np.dot(scaling_matrix, T)
# T = np.dot(translation_matrix, T)
# T = np.dot(rotation_matrix, T)
#
# dst3 = np.zeros((height, width, img.shape[2]), dtype=np.uint8)
#
# for y in range(height):
#     for x in range(width):
#
#         # 미리 구해놓은 변환행렬을 행렬곱 한번으로 적용합니다.
#         # 여기에서도 backward mapping을 사용합니다.
#         new_p = np.array([x, y, 1])
#         inv_scaling_matrix = np.linalg.inv(T)
#         old_p = np.dot(inv_scaling_matrix, new_p)
#
#         x_, y_ = old_p[:2]
#         x_ = int(x_)
#         y_ = int(y_)
#
#         if x_ > 0 and x_ < width and y_ > 0 and y_ < height:
#             dst3[y, x] = img[y_, x_]
# result2 = cv2.hconcat([result2,dst3])
# result = cv2.vconcat([result,result2])
#
# cv2.imshow("result", result)
# cv2.waitKey(0)

# 5번=======================================================================================
# import cv2
# import numpy as np
#
# img_src = cv2.imread('shape.png',cv2.IMREAD_COLOR)
# h,w=img_src.shape[:2]
# img_gray = cv2.cvtColor(img_src,cv2.COLOR_BGR2GRAY)
# img_gray=cv2.GaussianBlur(img_gray,(5,5),0)
# ret, img_binary = cv2.threshold(img_gray,230,255,cv2.THRESH_BINARY_INV)
# contours,hierarchy = cv2.findContours(img_binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
#
# # ===최대면적 area찾는 함수를위해 선언=========
# max_area = 0  # 최대 면적 초기화
# max_area_contour = None  # 최대 면적을 갖는 contour 초기화
#
# for i,contour in enumerate(contours):
#
#     area = cv2.contourArea(contour)
#     print( f"{i} = {area}")
#     if area>5000 :
#
#         mu = cv2.moments(contour)
#         cx = int(mu['m10'] / (mu['m00']+1e-5))
#         cy = int(mu['m01'] / (mu['m00'] + 1e-5))
#
#         x, y, w, h = cv2.boundingRect(contour)  # 주어진 윤곽선을 감싸는 사각형을 찾음.
#         cv2.drawContours(img_src,[contour], 0, (0,255,0),2)
#
#         # 모멘트(중심)에 텍스트 표시
#         cv2.circle((img_src),(cx,cy),5,(0,0,255),-1)
#         cv2.putText(img_src, f'{i} = {area:.0f}', (cx , cy+20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
#
#         # 최대 면적 갱신
#         if area > max_area:
#             max_area = area
#             max_area_contour = contour
#
#             MU = cv2.moments(max_area_contour)
#             CX = int(MU['m10'] / (MU['m00'] + 1e-5))
#             CY = int(MU['m01'] / (MU['m00'] + 1e-5))
#
#         cv2.imshow('src',img_src)
#
#         cv2.waitKey(0)
#         if cv2.waitKey(1) & 0xFF == 27: # ESC키
#             break
# if max_area_contour is not None:
#     X, Y, W, H = cv2.boundingRect(max_area_contour)  # 최대면적 도형 감싸는 사각형을 찾음.
#     cv2.rectangle(img_src, (X, Y, W, H), (0, 0, 255), 2)  # 최대면적 도형 빨간색 사각형으로 표시
#
# cv2.imshow('src',img_src)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # 6번 ===============================================================================
# import cv2
# import numpy as np
#
# # 트랙바 정의하고 이용하기
# def on_trackbar(x):
#     pass
#
# # trackbar에 대한 window를 별도로 생성
# cv2.namedWindow('Threshold Setting')
# cv2.resizeWindow('Threshold Setting', 400, 300)  # 필요에 따라 크기를 조절하세요
#
# # 해당 window(Threshold Setting)에 h_min1에 대한 trackbar 생성
# cv2.createTrackbar('h_min1', 'Threshold Setting', 0, 180, on_trackbar)
# cv2.setTrackbarPos('h_min1', 'Threshold Setting', 0)
#
# # 해당 window(Threshold Setting)에 h_max1에 대한 trackbar 생성
# cv2.createTrackbar('h_max1', 'Threshold Setting', 0, 180, on_trackbar)
# cv2.setTrackbarPos('h_max1', 'Threshold Setting', 34)
#
# # 해당 window(Threshold Setting)에 h_min2에 대한 trackbar 생성
# cv2.createTrackbar('h_min2', 'Threshold Setting', 0, 180, on_trackbar)
# cv2.setTrackbarPos('h_min2', 'Threshold Setting', 160)
#
# # 해당 window(Threshold Setting)에 h_max2에 대한 trackbar 생성
# cv2.createTrackbar('h_max2', 'Threshold Setting', 0, 180, on_trackbar)
# cv2.setTrackbarPos('h_max2', 'Threshold Setting', 180)
#
# # 해당 window(Threshold Setting)에 s_min에 대한 trackbar 생성
# cv2.createTrackbar('s_min', 'Threshold Setting', 0, 255, on_trackbar)
# cv2.setTrackbarPos('s_min', 'Threshold Setting', 147)
#
# # 해당 window(Threshold Setting)에 s_max에 대한 trackbar 생성
# cv2.createTrackbar('s_max', 'Threshold Setting', 0, 255, on_trackbar)
# cv2.setTrackbarPos('s_max', 'Threshold Setting', 247)
#
# # 해당 window(Threshold Setting)에 v_min에 대한 trackbar 생성
# cv2.createTrackbar('v_min', 'Threshold Setting', 0, 255, on_trackbar)
# cv2.setTrackbarPos('v_min', 'Threshold Setting', 77)
#
# # 해당 window(Threshold Setting)에 v_max에 대한 trackbar 생성
# cv2.createTrackbar('v_max', 'Threshold Setting', 0, 255, on_trackbar)
# cv2.setTrackbarPos('v_max', 'Threshold Setting', 253)
#
#
# while True:
#     img_src = cv2.imread('apple2.jpeg', cv2.IMREAD_COLOR)
#     height, width = img_src.shape[:2]
#     cv2.imshow('src', img_src)
#     img_hsv = cv2.cvtColor(img_src, cv2.COLOR_BGR2HSV)
#
#
#     h_min1 = cv2.getTrackbarPos('h_min1', 'Threshold Setting')
#     h_max1 = cv2.getTrackbarPos('h_max1', 'Threshold Setting')
#     h_min2 = cv2.getTrackbarPos('h_min2', 'Threshold Setting')
#     h_max2 = cv2.getTrackbarPos('h_max2', 'Threshold Setting')
#
#     s_min = cv2.getTrackbarPos('s_min', 'Threshold Setting')
#     s_max = cv2.getTrackbarPos('s_max', 'Threshold Setting')
#     v_min = cv2.getTrackbarPos('v_min', 'Threshold Setting')
#     v_max = cv2.getTrackbarPos('v_max', 'Threshold Setting')
#
#     # 설정 오류 검출을 위한 반복문 탈출
#     if s_max < s_min:
#         print("설정 오류입니다.")
#         break
#
#     if v_max < v_min:
#         print("설정 오류입니다.")
#         break
#
#     red_mask_low = cv2.inRange(img_hsv, (h_min1, s_min, v_min), (h_max1, s_max, v_max))
#     red_mask_high = cv2.inRange(img_hsv, (h_min2, s_min, v_min), (h_max2, s_max, v_max))
#
#     red_mask = cv2.addWeighted(red_mask_low, 1.0, red_mask_high, 1.0, 0.0)
#     img_red = cv2.bitwise_and(img_hsv, img_hsv, mask=red_mask)
#     img_red = cv2.cvtColor(img_red, cv2.COLOR_HSV2BGR)
#
#     cv2.imshow('src', img_red)
#     cv2.imshow('h', red_mask)
#
#     img_red_gray = cv2.cvtColor(img_red, cv2.COLOR_BGR2GRAY)
#     contours, hierarchy = cv2.findContours(img_red_gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
#
#     for i, contour in enumerate(contours):
#         area = cv2.contourArea(contour)
#         if area > 20000:
#             mu = cv2.moments(contour)
#             cx = int(mu['m10'] / (mu['m00'] + 1e-5))
#             cy = int(mu['m01'] / (mu['m00'] + 1e-5))
#             cv2.circle(img_red, (cx, cy), 3, (0, 0, 0), -1)
#
#             # 사각형 그리기
#             # cv2.drawContours(img_src, [contour], 0, (0, 0, 0), 2)
#             x, y, w, h = cv2.boundingRect(contour)
#             cv2.rectangle(img_src, (x, y, w, h), (0, 255, 255), 2)
#             # # # areaR = cv2.Area
#             # # # 사각형 그리기 끝
#
#             cv2.circle(img_src,(cx , cy ),10,(0,0,255),-1)
#             cv2.putText(img_src, f' Area : {area:.2f}', (cx - 120, cy + 50),
#             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
#
#             cv2.imshow("final result",img_src)
#
#     # 키보드 입력 대기 시간 : 1ms / ESC(27) 누를시 종료
#     key = cv2.waitKey(1)
#     if key == 27:
#         break
#
# cv2.destroyAllWindows()


# 231129 자습 ================동영상캡쳐 함수로 사진 불러오기
# import cv2
#
# img=cv2.imread("love.png",cv2.IMREAD_COLOR)
#
# while cv2.waitKey(3)<0:
#     ret,img_frame =capture.read()
#     if ret ==False:
#         print("캡쳐실패")
#         break
#
#     cv2.imshow('Color',img_frame)
#
#     key=cv2.waitKey(3)
#     if key==30:
#         break
# capture.release()
# cv2.destroyAllWindows()

# ============231129 혼자연습계층관계면 안에 요소 다 없애기
# # =================컨터(외곽선)채우기 ===============================================================
# import cv2
#
# img_src = cv2.imread('coin.png',cv2.IMREAD_COLOR)
# height,width = img_src.shape[:2]
# # color to gray
# img_gray = cv2.cvtColor(img_src,cv2.COLOR_BGR2GRAY)
# # smoothing(bluring)
# img_gray = cv2.GaussianBlur(img_gray,(15,15), 0)
# img_binary = cv2.adaptiveThreshold(img_gray,255,
#                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                    cv2.THRESH_BINARY_INV, 11, 1)
# ################# 모폴로지 ####################
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
# img_binary = cv2.morphologyEx(img_binary,cv2.MORPH_OPEN,kernel,iterations=2)
# img_binary = cv2.morphologyEx(img_binary,cv2.MORPH_CLOSE,kernel,iterations=2)
# #############################################
# contours, hierarchy = cv2.findContours(img_binary,
#                                        cv2.RETR_CCOMP, # 계층 존재 2단계 아래 삭제( 계층없으면 " cv2.RETR_LIST ",)
#                                        cv2.CHAIN_APPROX_NONE)
# for i,contour in enumerate(contours):
#     print(i, hierarchy[0][i])
#     # print(i, hierarchy[0][i][2])
#     #if hierarchy[0][i][2] != -1:
#
#     cv2.drawContours(img_src,[contour],0,(0,255,0),2)
#     cv2.putText(img_src,str(i),tuple(contour[0][0]),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),1)
#
# cv2.imshow('src',img_src)
# cv2.imshow('dst',img_binary)
# # ======================오픈씨브이 보충수업 사진저장(롸이트 함수)
# cv2.imwrite('result.png',img_binary)
#
# cv2.waitKey()
# cv2.destroyAllWindows()


# =================231129 문제 4번 다시 코드 더 짧게 해서 그림을 계속 이어받아서? 풀어보기
# import cv2
# import numpy as np
#
# img = cv2.imread('lion.png', cv2.IMREAD_COLOR)
# img=cv2.pyrDown(img)    #사진 너무커서 사이즈 줄이고 시작.
# height, width = img.shape[:2]
#
# # 원본과 0.7배 합치기
# scale_factor = 0.7
# scaling_matrix = np.array([[scale_factor, 0, 0], [0, scale_factor, 0], [0, 0, 1]])
# T = np.eye(3)
# T = np.dot(scaling_matrix, T)
#
# dst = np.zeros((height, width, img.shape[2]), dtype=np.uint8)
#
# for y in range(height):
#     for x in range(width):
#
#         # 미리 구해놓은 변환행렬을 행렬곱 한번으로 적용합니다.
#         # 여기에서도 backward mapping을 사용합니다.
#         new_p = np.array([x, y, 1])
#         inv_scaling_matrix = np.linalg.inv(T)
#         old_p = np.dot(inv_scaling_matrix, new_p)
#
#         x_, y_ = old_p[:2]
#         x_ = int(x_)
#         y_ = int(y_)
#
#         if 0 <= x_ < width and 0 <= y_ < height:
#             dst[y, x] = img[y_, x_]
# # result = cv2.hconcat([img,dst])  # 1번,2번 가로로 합성완료
#
#
# # 15%이동
# translation_matrix = np.array([[1, 0, width*0.15], [0, 1,height*0.15], [0, 0, 1]])
#
# # 정해진 순서대로 변환 행렬을 곱하여 하나의 행렬을 생성합니다.
# T = np.eye(3)
# T = np.dot(translation_matrix, T)
#
# dst2 = np.zeros((height, width, img.shape[2]), dtype=np.uint8)
#
# for y in range(height):
#     for x in range(width):
#
#         # 미리 구해놓은 변환행렬을 행렬곱 한번으로 적용합니다.
#         # 여기에서도 backward mapping을 사용합니다.
#         new_p = np.array([x, y, 1])
#         inv_scaling_matrix = np.linalg.inv(T)
#         old_p = np.dot(inv_scaling_matrix, new_p)
#
#         x_, y_ = old_p[:2]
#         x_ = int(x_)
#         y_ = int(y_)
#
#         if 0 <= x_ < width and 0 <= y_ < height:
#             dst2[y, x] = dst[y_, x_]
#
# # 45도 회전 한걸  "15%이동한사진" 과 붙이기
# translation_matrix = np.array([[1, 0, width*(1-scale_factor)/2], [0, 1,height*(1-scale_factor)/2], [0, 0, 1]])
#                                                                     # 회전중심 중앙 맞추기
# angle = 45
# radian = angle * np.pi / 180
# c = np.cos(radian)
# s = np.sin(radian)
# center_x = width / 2
# center_y = height / 2
# rotation_matrix = np.array(
#     [[c, s, (1 - c) * center_x - s * center_y], [-s, c, s * center_x + (1 - c) * center_y], [0, 0, 1]])
#
# # 정해진 순서대로 변환 행렬을 곱하여 하나의 행렬을 생성합니다.
# T = np.eye(3)
# T = np.dot(rotation_matrix, T)
#
# dst3 = np.zeros((height, width, img.shape[2]), dtype=np.uint8)
#
# for y in range(height):
#     for x in range(width):
#
#         # 미리 구해놓은 변환행렬을 행렬곱 한번으로 적용합니다.
#         # 여기에서도 backward mapping을 사용합니다.
#         new_p = np.array([x, y, 1])
#         inv_scaling_matrix = np.linalg.inv(T)
#         old_p = np.dot(inv_scaling_matrix, new_p)
#
#         x_, y_ = old_p[:2]
#         x_ = int(x_)
#         y_ = int(y_)
#
#         if x_ > 0 and x_ < width and y_ > 0 and y_ < height:
#             dst3[y, x] = dst2[y_, x_]
# result = cv2.hconcat([img,dst,dst2,dst3])
#
# cv2.imshow("result", result)
# cv2.waitKey(0)

# ===============231129 문제5번 다시 풀어보기

# import cv2
# import numpy as np
#
# img_src = cv2.imread('shape_exam')





#  ==================231205
# # =======231205사진 16조각내고 모서리 패딩하고  폴더 두개에 반반씩 집어넣기.
# import cv2
# import os
#
# img_src = cv2.imread("lion.png", cv2.IMREAD_COLOR)
# h, w = img_src.shape[:2]
#
# # 폴더 두개 만들기
# folder_path = 'A1'
# os.makedirs(folder_path, exist_ok=True)  # 폴더 100개 생성완료
# folder_path = 'A2'
# os.makedirs(folder_path, exist_ok=True)  # 폴더 100개 생성완료
# img_src_border = img_src.copy()
# if h % 16 != 0:
#     if (h % 16) % 2 == 0:  # 짝수라면,
#         y = (16 - (h % 16)) // 2  # 상하 패드 갯수는 각각 y개
#         img_src_border = cv2.copyMakeBorder(img_src_border, y, y, 0, 0, cv2.BORDER_CONSTANT)
#     if (w % 16) % 2 == 0:  # 짝수라면,
#         x = (16 - (h % 16)) // 2  # 좌우 패드 갯수는 각각 x개
#         img_src_border = cv2.copyMakeBorder(img_src_border, 0, 0, x, x, cv2.BORDER_CONSTANT)
#
#     if (h % 16) % 2 == 1:  # 홀수 라면,
#         y = (16 - (h % 16)) // 2  # 상하 패드 갯수는 각각 y, y+1개
#         img_src_border = cv2.copyMakeBorder(img_src_border, y, y + 1, 0, 0, cv2.BORDER_CONSTANT)
#     if (w % 16) % 2 == 1:  # 홀수라면,
#         x = (16 - (h % 16)) // 2  # 좌우 패드 갯수는 각각 x, x+1개
#         img_src_border = cv2.copyMakeBorder(img_src_border, 0, 0, x, x + 1, cv2.BORDER_CONSTANT)
#
# h, w = img_src_border.shape[:2]
# hh = h // 16
# ww = w // 16
#
# # 위 / 아래로 나눠서 파일 넣기.
# # for j in range(0,8):
# #     for i in range(0,16):
# #         img_puzzle=img_src_border[hh*j:hh*(j+1),ww*i:ww*(i+1)]
# #         file_path = os.path.join("A1", f'puzzle{j}_{i}.png')
# #         cv2.imwrite(file_path, img_puzzle)
# # for j in range(8,16):
# #     for i in range(0,16):
# #         img_puzzle=img_src_border[hh*j:hh*(j+1),ww*i:ww*(i+1)]
# #         file_path = os.path.join("A2", f'puzzle{j}_{i}.png')
# #         cv2.imwrite(file_path, img_puzzle)
#
# # 좌/우로 나눠서 파일 넣기.
# for j in range(0,16):
#     for i in range(0,8):
#         img_puzzle=img_src_border[hh*j:hh*(j+1),ww*i:ww*(i+1)]
#         file_path = os.path.join("A1", f'puzzle{j}_{i}.png')
#         cv2.imwrite(file_path, img_puzzle)
# for j in range(0,16):
#     for i in range(8,16):
#         img_puzzle=img_src_border[hh*j:hh*(j+1),ww*i:ww*(i+1)]
#         file_path = os.path.join("A2", f'puzzle{j}_{i}.png')
#         cv2.imwrite(file_path, img_puzzle)
#
#
#
#
# cv2.imshow('src', img_src)
# cv2.imshow('dst', img_src_border)
#
# cv2.waitKey()
# cv2.destroyAllWindows()



# # 파일 다시 합치기.
# import cv2
# import os
# import numpy as np
#
#
#
#
#
# for j in range(0,16):
#     for i in range(0,7):
#         img_puzzle_p1 = cv2.imread(f"./A1/puzzle{j}_{i}.png", cv2.IMREAD_COLOR)
#         img_puzzle_p2 = cv2.imread(f"./A1/puzzle{j}_{i+1}.png", cv2.IMREAD_COLOR)
#         if i==0:
#             img_puzzle_p = np.concatenate((img_puzzle_p1, img_puzzle_p2), axis=1)
#         else:
#             img_puzzle_p = np.concatenate((img_puzzle_p, img_puzzle_p2), axis=1)
#
#     if j==0:
#         img_puzzle0_finish =img_puzzle_p.copy()
#         img_puzzle_p=None
#     if j==1:
#         img_puzzle1_finish =img_puzzle_p.copy()
#         img_puzzle_p=None
#     if j==2:
#         img_puzzle2_finish =img_puzzle_p.copy()
#         img_puzzle_p=None
#     if j==3:
#         img_puzzle3_finish =img_puzzle_p.copy()
#         img_puzzle_p=None
#     if j==4:
#         img_puzzle4_finish =img_puzzle_p.copy()
#         img_puzzle_p=None
#     if j==5:
#         img_puzzle5_finish =img_puzzle_p.copy()
#         img_puzzle_p=None
#     if j==6:
#         img_puzzle6_finish =img_puzzle_p.copy()
#         img_puzzle_p=None
#     if j==7:
#         img_puzzle7_finish =img_puzzle_p.copy()
#         img_puzzle_p=None
#     if j==8:
#         img_puzzle8_finish =img_puzzle_p.copy()
#         img_puzzle_p=None
#     if j==9:
#         img_puzzle9_finish =img_puzzle_p.copy()
#         img_puzzle_p=None
#     if j==10:
#         img_puzzle10_finish =img_puzzle_p.copy()
#         img_puzzle_p=None
#     if j==11:
#         img_puzzle11_finish =img_puzzle_p.copy()
#         img_puzzle_p=None
#     if j==12:
#         img_puzzle12_finish =img_puzzle_p.copy()
#         img_puzzle_p=None
#     if j==13:
#         img_puzzle13_finish =img_puzzle_p.copy()
#         img_puzzle_p=None
#     if j==14:
#         img_puzzle14_finish =img_puzzle_p.copy()
#         img_puzzle_p=None
#     if j==15:
#         img_puzzle15_finish =img_puzzle_p.copy()
#         img_puzzle_p=None
#         img_left = np.concatenate((img_puzzle0_finish,img_puzzle1_finish,img_puzzle2_finish,img_puzzle3_finish,img_puzzle4_finish,
#             img_puzzle5_finish,img_puzzle6_finish,img_puzzle7_finish,img_puzzle8_finish,img_puzzle9_finish,
#             img_puzzle10_finish,img_puzzle11_finish,img_puzzle12_finish,img_puzzle13_finish,img_puzzle14_finish,img_puzzle15_finish), axis=0)
# for j in range(0,16):
#     for i in range(8,15):
#         img_puzzle_p1 = cv2.imread(f"./A2/puzzle{j}_{i}.png", cv2.IMREAD_COLOR)
#         img_puzzle_p2 = cv2.imread(f"./A2/puzzle{j}_{i+1}.png", cv2.IMREAD_COLOR)
#         if i==8:
#             img_puzzle_p = np.concatenate((img_puzzle_p1, img_puzzle_p2), axis=1)
#         else:
#             img_puzzle_p = np.concatenate((img_puzzle_p, img_puzzle_p2), axis=1)
#
#     if j==0:
#         img_puzzle0_finish =img_puzzle_p.copy()
#         img_puzzle_p=None
#
#     if j==1:
#         img_puzzle1_finish =img_puzzle_p.copy()
#         img_puzzle_p=None
#     if j==2:
#         img_puzzle2_finish =img_puzzle_p.copy()
#         img_puzzle_p=None
#     if j==3:
#         img_puzzle3_finish =img_puzzle_p.copy()
#         img_puzzle_p=None
#     if j==4:
#         img_puzzle4_finish =img_puzzle_p.copy()
#         img_puzzle_p=None
#     if j==5:
#         img_puzzle5_finish =img_puzzle_p.copy()
#         img_puzzle_p=None
#     if j==6:
#         img_puzzle6_finish =img_puzzle_p.copy()
#         img_puzzle_p=None
#     if j==7:
#         img_puzzle7_finish =img_puzzle_p.copy()
#         img_puzzle_p=None
#     if j==8:
#         img_puzzle8_finish =img_puzzle_p.copy()
#         img_puzzle_p=None
#     if j==9:
#         img_puzzle9_finish =img_puzzle_p.copy()
#         img_puzzle_p=None
#     if j==10:
#         img_puzzle10_finish =img_puzzle_p.copy()
#         img_puzzle_p=None
#     if j==11:
#         img_puzzle11_finish =img_puzzle_p.copy()
#         img_puzzle_p=None
#     if j==12:
#         img_puzzle12_finish =img_puzzle_p.copy()
#         img_puzzle_p=None
#     if j==13:
#         img_puzzle13_finish =img_puzzle_p.copy()
#         img_puzzle_p=None
#     if j==14:
#         img_puzzle14_finish =img_puzzle_p.copy()
#         img_puzzle_p=None
#     if j==15:
#         img_puzzle15_finish =img_puzzle_p.copy()
#         img_puzzle_p=None
#         img_right = np.concatenate((img_puzzle0_finish,img_puzzle1_finish,img_puzzle2_finish,img_puzzle3_finish,img_puzzle4_finish,
#             img_puzzle5_finish,img_puzzle6_finish,img_puzzle7_finish,img_puzzle8_finish,img_puzzle9_finish,
#             img_puzzle10_finish,img_puzzle11_finish,img_puzzle12_finish,img_puzzle13_finish,img_puzzle14_finish,img_puzzle15_finish), axis=0)
#
# img_left = np.concatenate((img_left,img_right),axis=1)
#
# img_repair = np.zeros((600, 600, 3), dtype=np.uint8)
#
# img_repair[0:600, 0:600, :] = img_left[4:604, 4:604, :]
#
# img_src = cv2.imread("lion.png",cv2.IMREAD_COLOR)
#
# cv2.imshow('repair', img_repair)
# cv2.imshow('left', img_left)
# cv2.imshow('src', img_src)
#
# cv2.waitKey()
# cv2.destroyAllWindows()



#  ======================231205 레인 디텍팅=============================================#################################################

# 필요한 몇 가지 패키지를 가져오기
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os

# 모든 이미지 가져오기
test_images = [mpimg.imread('test_video2/videoFrames/' + i) for i in os.listdir('test_video2/videoFrames/')]

# ----------------이미지 가져오기-----------------
im = test_images[0]
imshape = im.shape
h,w = im.shape[:2]   # h, w 정의==================================================================
# -------비디오 라이터 생성------------------
# 코덱 정의 및 VideoWriter 객체 생성
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
videoOut = cv2.VideoWriter('output2.avi', fourcc, 20.0, (im.shape[1], im.shape[0]))

# 트랙바 정의하고 이용하기===============================================================================



def on_trackbar(x):
     pass


# trackbar에 대한 window를 별도로 생성==========================================================
cv2.namedWindow('trapezoid Setting')
cv2.resizeWindow('trapezoid Setting', 800, 300)  # 필요에 따라 크기를 조절하세요

# 해당 window(trapezoid Setting)에 L_U_x에 대한 trackbar 생성=============================================
cv2.createTrackbar('L_U_x', 'trapezoid Setting', 0, w, on_trackbar)
cv2.setTrackbarPos('L_U_x', 'trapezoid Setting', 465)
# 해당 window(trapezoid Setting)에 L_U_y에 대한 trackbar 생성
cv2.createTrackbar('L_U_y', 'trapezoid Setting', 0, h, on_trackbar)
cv2.setTrackbarPos('L_U_y', 'trapezoid Setting', 320)
# 해당 window(trapezoid Setting)에 R_U_x에 대한 trackbar 생성
cv2.createTrackbar('R_U_x', 'trapezoid Setting', 0, w, on_trackbar)
cv2.setTrackbarPos('R_U_x', 'trapezoid Setting', 475)
# 해당 window(trapezoid Setting)에 R_U_y에 대한 trackbar 생성
cv2.createTrackbar('R_U_y', 'trapezoid Setting', 0, h, on_trackbar)
cv2.setTrackbarPos('R_U_y', 'trapezoid Setting', 320)

cv2.createTrackbar('minVal', 'trapezoid Setting', 0, 255, on_trackbar)
cv2.setTrackbarPos('minVal', 'trapezoid Setting', 60)
cv2.createTrackbar('maxVal', 'trapezoid Setting', 0, 255, on_trackbar)
cv2.setTrackbarPos('maxVal', 'trapezoid Setting', 150)

minVal = 60
maxVal = 150

for frameNum in range(1, len(test_images) - 1):

    if frameNum % 50 == 0:
        print('프레임 번호: ', frameNum)

    im = mpimg.imread('test_video2/videoFrames/' + str(frameNum) + '.jpg')
    imshape = im.shape

    # -------------그레이스케일 이미지---------------
    grayIm = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # ------------가우시안 스무딩-----------------
    kernel_size = 9
    smoothedIm = cv2.GaussianBlur(grayIm, (kernel_size, kernel_size), 0)

    # -------------에지 검출---------------------

    edgesIm = cv2.Canny(smoothedIm, minVal, maxVal)
    cv2.imshow("edgesIm ", edgesIm)
    cv2.waitKey(1)

    # -------------------------트랙바 생성(실시간 갱신)--------------------------------=================================
    L_U_x = cv2.getTrackbarPos('L_U_x', 'trapezoid Setting')
    L_U_y = cv2.getTrackbarPos('L_U_y', 'trapezoid Setting')
    R_U_x = cv2.getTrackbarPos('R_U_x', 'trapezoid Setting')
    R_U_y = cv2.getTrackbarPos('R_U_y', 'trapezoid Setting')
    minVal = cv2.getTrackbarPos('minVal', 'trapezoid Setting')
    maxVal = cv2.getTrackbarPos('maxVal', 'trapezoid Setting')

    # -------------------------마스크 생성--------------------------------
    vertices = np.array([[(0, imshape[0]), (L_U_x, L_U_y), (R_U_x, R_U_y), (imshape[1], imshape[0])]], dtype=np.int32)
    mask = np.zeros_like(edgesIm)
    color = 255
    cv2.fillPoly(mask, vertices, color)
    # 실시간 모니터링하기 =========================================================================================
    cv2.imshow("mask",mask)
    cv2.waitKey(33)

    # ----------------------이미지에 마스크 적용-------------------------------
    maskedIm = cv2.bitwise_and(edgesIm, mask)   # 마스크영역에만 엣지(흑색화면에 흰엣지) 처리함
    maskedIm3Channel = cv2.cvtColor(maskedIm, cv2.COLOR_GRAY2BGR)  # 3채널 모드로 디멘전만 변경
    # 실시간 모니터링하기 =========================================================================================
    cv2.imshow("maskedIm",maskedIm)
    cv2.imshow("maskedIm3Channel", maskedIm3Channel)
    cv2.waitKey(33)

    # -----------------------허프 라인 검출------------------------------------
    rho = 2
    theta = np.pi / 180
    threshold = 45
    min_line_len = 40
    max_line_gap = 100
    lines = cv2.HoughLinesP(maskedIm, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len, maxLineGap=max_line_gap)

    # 라인이 2개 이상인지 확인
    if lines is not None and len(lines) > 2:

        # 모든 라인을 이미지에 그리기
        allLines = np.zeros_like(maskedIm)
        for i in range(len(lines)):
            for x1, y1, x2, y2 in lines[i]:
                cv2.line(allLines, (x1, y1), (x2, y2), (255, 255, 0), 2)  # 라인 그리기

        # -----------------------양/음의 기울기를 가진 라인 분리--------------------------
        slopePositiveLines = []  # x1 y1 x2 y2 slope
        slopeNegativeLines = []
        yValues = []

        addedPos = False
        addedNeg = False
        for currentLine in lines:
            for x1, y1, x2, y2 in currentLine:
                lineLength = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** .5  # 라인 길이 계산
                if lineLength > 30:  # 라인이 충분히 길 경우
                    if x2 != x1:
                        slope = (y2 - y1) / (x2 - x1)
                        if slope > 0:
                            tanTheta = np.tan((abs(y2 - y1)) / (abs(x2 - x1)))
                            ang = np.arctan(tanTheta) * 180 / np.pi
                            if abs(ang) < 85 and abs(ang) > 20:
                                slopeNegativeLines.append([x1, y1, x2, y2, -slope])
                                yValues.append(y1)
                                yValues.append(y2)
                                addedPos = True
                        if slope < 0:
                            tanTheta = np.tan((abs(y2 - y1)) / (abs(x2 - x1)))
                            ang = np.arctan(tanTheta) * 180 / np.pi
                            if abs(ang) < 85 and abs(ang) > 20:
                                slopePositiveLines.append([x1, y1, x2, y2, -slope])
                                yValues.append(y1)
                                yValues.append(y2)
                                addedNeg = True

        if not addedPos or not addedNeg:
            print('충분한 라인이 발견되지 않았습니다.')

        positiveSlopes = np.asarray(slopePositiveLines)[:, 4]
        posSlopeMedian = np.median(positiveSlopes)  # 중앙값
        posSlopeStdDev = np.std(positiveSlopes)
        posSlopesGood = []
        for slope in positiveSlopes:
            if abs(slope - posSlopeMedian) < posSlopeMedian * .2:
                posSlopesGood.append(slope)         # 검열을 모두 통과한 이상적인라인들 검출
        posSlopeMean = np.mean(np.asarray(posSlopesGood))

        negativeSlopes = np.asarray(slopeNegativeLines)[:, 4]
        negSlopeMedian = np.median(negativeSlopes)
        negSlopeStdDev = np.std(negativeSlopes)
        negSlopesGood = []
        for slope in negativeSlopes:
            if abs(slope - negSlopeMedian) < .9:
                negSlopesGood.append(slope)
        negSlopeMean = np.mean(np.asarray(negSlopesGood))

        xInterceptPos = []
        for line in slopePositiveLines:
            x1 = line[0]
            y1 = im.shape[0] - line[1]
            slope = line[4]
            yIntercept = y1 - slope * x1
            xIntercept = -yIntercept / slope
            if xIntercept == xIntercept:
                xInterceptPos.append(xIntercept)

        xIntPosMed = np.median(xInterceptPos)
        xIntPosGood = []
        for line in slopePositiveLines:
            x1 = line[0]
            y1 = im.shape[0] - line[1]
            slope = line[4]
            yIntercept = y1 - slope * x1
            xIntercept = -yIntercept / slope
            if abs(xIntercept - xIntPosMed) < .35 * xIntPosMed:
                xIntPosGood.append(xIntercept)

        xInterceptPosMean = np.mean(np.asarray(xIntPosGood))

        xInterceptNeg = []
        for line in slopeNegativeLines:
            x1 = line[0]
            y1 = im.shape[0] - line[1]
            slope = line[4]
            yIntercept = y1 - slope * x1
            xIntercept = -yIntercept / slope
            if xIntercept == xIntercept:
                xInterceptNeg.append(xIntercept)

        xIntNegMed = np.median(xInterceptNeg)
        xIntNegGood = []
        for line in slopeNegativeLines:
            x1 = line[0]
            y1 = im.shape[0] - line[1]
            slope = line[4]
            yIntercept = y1 - slope * x1
            xIntercept = -yIntercept / slope
            if abs(xIntercept - xIntNegMed) < .35 * xIntNegMed:
                xIntNegGood.append(xIntercept)

        xInterceptNegMean = np.mean(np.asarray(xIntNegGood))

    # ----------------------차선 라인 플로팅------------------------------
    laneLines = np.zeros_like(edgesIm)
    colorLines = im.copy()

    slope = posSlopeMean
    x1 = xInterceptPosMean
    y1 = 0
    y2 = imshape[0] - (imshape[0] - imshape[0] * .35)
    x2 = (y2 - y1) / slope + x1

    x1 = int(round(x1))
    x2 = int(round(x2))
    y1 = int(round(y1))
    y2 = int(round(y2))

    jumpThresh = 50
    if 'x1Prev' in globals():
        if abs(x1 - x1Prev) > 3 and abs(x1 - x1Prev) < jumpThresh:
            x1 = x1Prev + np.sign(x1 - x1Prev) * 1
        if abs(x2 - x2Prev) > 3 and abs(x2 - x2Prev) < jumpThresh:
            x2 = x2Prev + np.sign(x2 - x2Prev) * 1

    cv2.line(laneLines, (x1, im.shape[0] - y1), (x2, imshape[0] - y2), (255, 255, 0), 2)
    cv2.line(colorLines, (x1, im.shape[0] - y1), (x2, imshape[0] - y2), (0, 255, 0), 4)

    slope = negSlopeMean
    x1N = xInterceptNegMean
    y1N = 0
    x2N = (y2 - y1N) / slope + x1N

    x1N = int(round(x1N))
    x2N = int(round(x2N))
    y1N = int(round(y1N))

    if 'x1NPrev' in globals():
        if abs(x1N - x1NPrev) > 3 and abs(x1N - x1NPrev) < jumpThresh:
            x1N = x1NPrev + np.sign(x1N - x1NPrev) * 1
        if abs(x2N - x2NPrev) > 3 and abs(x2N - x2NPrev) < jumpThresh:
            x2N = x2NPrev + np.sign(x2N - x2NPrev) * 1

    cv2.line(laneLines, (x1N, imshape[0] - y1N), (x2N, imshape[0] - y2), (255, 255, 0), 2)
    cv2.line(colorLines, (x1N, im.shape[0] - y1N), (x2N, imshape[0] - y2), (0, 255, 0), 4)

    x1Prev = x1
    x2Prev = x2
    y1Prev = y1
    y2Prev = y2
    x1NPrev = x1N
    x2NPrev = x2N
    y1NPrev = y1N

    laneFill = im.copy()
    vertices = np.array([[(x1, im.shape[0] - y1), (x2, im.shape[0] - y2), (x2N, imshape[0] - y2),
                          (x1N, imshape[0] - y1N)]], dtype=np.int32)
    color = [241, 255, 1]
    cv2.fillPoly(laneFill, vertices, color)
    opacity = .25
    blendedIm = cv2.addWeighted(laneFill, opacity, im, 1 - opacity, 0, im)
    cv2.line(blendedIm, (x1, im.shape[0] - y1), (x2, imshape[0] - y2), (0, 255, 0), 4)
    cv2.line(blendedIm, (x1N, im.shape[0] - y1N), (x2N, imshape[0] - y2), (0, 255, 0), 4)
    b, g, r = cv2.split(blendedIm)
    outputIm = cv2.merge((r, g, b))

    cv2.imshow("vid", outputIm)
    videoOut.write(outputIm)

videoOut.release()