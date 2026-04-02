def outer_func():
    a = 20  # 2. outer_func의 local variable 생성
    def inner_func():  # 3. inner_func는 여기서 정의만 되고 아직 실행되지는 않음
        a = 30  # 5. inner_func의 local variable 생성
        print(locals())  # 6. inner_func의 locals() 출력 -> {'a': 30}

    inner_func()  # 4. inner_func 실행
    print(locals())  # 7. outer_func의 locals() 출력 -> {'a': 20, 'inner_func': <function ...>}

a = 10  # 1. global variable 생성
outer_func()  # global scope에서 outer_func 호출
print(locals())  # 8. global scope의 locals() 출력 -> {..., 'outer_func': <function ...>, 'a': 10}

# >> a = 30
# >> a = 20
# >> a = 10

#inner_func, locals(): #{'a': 30} 
#outer_func, locals(): #{'a': 20, 'inner_func': <function outer_func.<locals>.inner_func at 0x72d8c4d2a200>}
#global scope, locals(): #{'__name__': '__main__', '__doc__': None, '__package__': None, '__loader__': <_frozen_importlib_external.SourceFileLoader object at 0x72d8c4cb9c30>, '__spec__': None, '__annotations__': {}, '__builtins__': <module 'builtins' (built-in)>, '__file__': '/home/mhlee/Documents/High_Performance_Data_Analysis/week03/namespace.py', '__cached__': None, 'outer_func': <function outer_func at 0x72d8c509fd90>, 'a': 10}{'__name__': '__main__', '__doc__': None, '__package__': None, '__loader__': <_frozen_importlib_external.SourceFileLoader object at 0x71708351dc30>, '__spec__': None, '__annotations__': {}, '__builtins__': <module 'builtins' (built-in)>, '__file__': '/home/mhlee/Documents/High_Performance_Data_Analysis/week03/namespace.py', '__cached__': None, 'outer_func': <function outer_func at 0x717083923d90>, 'a': 10}
