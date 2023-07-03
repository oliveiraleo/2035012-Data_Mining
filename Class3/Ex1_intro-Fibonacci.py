def fib(n = 10):
    "Calculates fibonacci"
    numbers = []
    i = 0
    for i in range(n):
        if i == 0:
            numbers.append(0)
        elif i == 1:
            numbers.append(1)
        else:
            last_two_on_tail = numbers[-2:]
            next_num_of_seq = last_two_on_tail[0] + last_two_on_tail[1]
            numbers.append(next_num_of_seq)
    
    print("The first %i elements of the fibonacci sequence are:" % (n))
    print(numbers)

print("Please, enter how many elements of the fibonacci sequence you want to calculate")
user_input = input("> ")
fib(int(user_input))