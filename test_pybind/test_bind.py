import module

#Test 1 addition
print('====------Addition-----====')
res_add = module.add(3,4)
print("3+4 = ", res_add)
print('Returned type ', type(res_add))


print('====----Area of a rectanfle----====')
res_area = module.area_rect(7,4)
print("Area of the rectangle with length = 7 and width = 4 ",res_area)
print('Returned type', type(res_area))

print('====----Saying Hello----====')
res_hello = module.hello('Aziz')
print(res_hello)
print('Returned type ', type(res_hello))
# print(module.__doc__)
# help(module.add)

print("Error testing")
try:
    module.add("Hi", 5)
except TypeError as e:
    print(f"Error intercepted: {e}")