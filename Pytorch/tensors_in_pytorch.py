import torch
print(torch.__version__)

if torch.cuda.is_available():
    print("GPU is available!")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("GPU not available. Using CPU.")

# using empty
a = torch.empty(2,3)

# check type
type(a)

# using zeros
torch.zeros(2,3)

# using ones
torch.ones(2,3)

# using rand
torch.rand(2,3)

# use of seed
torch.rand(2,3)

# manual_seed
torch.manual_seed(100)
torch.rand(2,3)

torch.manual_seed(100)
torch.rand(2,3)

# using tensor
torch.tensor([[1,2,3],[4,5,6]])

# other ways

# arange
print("using arange ->", torch.arange(0,10,2))

# using linspace
print("using linspace ->", torch.linspace(0,10,10))

# using eye
print("using eye ->", torch.eye(5))

# using full
print("using full ->", torch.full((3, 3), 5))

x = torch.tensor([[1,2,3],[4,5,6]])
x

x.shape

torch.empty_like(x)

torch.zeros_like(x)

torch.ones_like(x)

torch.rand_like(x, dtype=torch.float32)

# find data type
x.dtype

# assign data type
torch.tensor([1.0,2.0,3.0], dtype=torch.int32)

# using to()
x.to(torch.float32)

x = torch.rand(2,2)
x

# addition
x + 2
# substraction
x - 2
# multiplication
x * 3
# division
x / 3
# int division
(x * 100)//3
# mod
((x * 100)//3)%2
# power
x**2

a = torch.rand(2,3)
b = torch.rand(2,3)

print(a)
print(b)

# add
a + b
# sub
a - b
# multiply
a * b
# division
a / b
# power
a ** b
# mod
a % b

c = torch.tensor([1, -2, 3, -4])

# abs
torch.abs(c)

# negative
torch.neg(c)

d = torch.tensor([1.9, 2.3, 3.7, 4.4])

# round
torch.round(d)

# ceil
torch.ceil(d)

# floor
torch.floor(d)

# clamp
torch.clamp(d, min=2, max=3)

e = torch.randint(size=(2,3), low=0, high=10, dtype=torch.float32)
e

# sum
torch.sum(e)
# sum along columns
torch.sum(e, dim=0)
# sum along rows
torch.sum(e, dim=1)

# mean
torch.mean(e)
# mean along col
torch.mean(e, dim=0)

# median
torch.median(e)

# max and min
torch.max(e)
torch.min(e)

# product
torch.prod(e)

# standard deviation
torch.std(e)

# variance
torch.var(e)

# argmax
torch.argmax(e)

# argmin
torch.argmin(e)

f = torch.randint(size=(2,3), low=0, high=10)
g = torch.randint(size=(3,2), low=0, high=10)

print(f)
print(g)

# matrix multiplcation
torch.matmul(f, g)

vector1 = torch.tensor([1, 2])
vector2 = torch.tensor([3, 4])

# dot product
torch.dot(vector1, vector2)

# transpose
torch.transpose(f, 0, 1)

h = torch.randint(size=(3,3), low=0, high=10, dtype=torch.float32)
h

# determinant
torch.det(h)

# inverse
torch.inverse(h)

i = torch.randint(size=(2,3), low=0, high=10)
j = torch.randint(size=(2,3), low=0, high=10)

print(i)
print(j)

# greater than
i > j
# less than
i < j
# equal to
i == j
# not equal to
i != j
# greater than equal to

# less than equal to

k = torch.randint(size=(2,3), low=0, high=10, dtype=torch.float32)
k

# log
torch.log(k)

# exp
torch.exp(k)

# sqrt
torch.sqrt(k)

# sigmoid
torch.sigmoid(k)

# softmax
torch.softmax(k, dim=0)

# relu
torch.relu(k)

m = torch.rand(2,3)
n = torch.rand(2,3)

print(m)
print(n)

m.add_(n)
m
n
torch.relu(m)
m.relu_()
m
a = torch.rand(2,3)
a

b = a
b
a[0][0] = 0
a
b

id(a)
id(b)
a
b
a[0][0] = 10
a
b
id(a)
id(b)