# this script computes the factorial of the given number using recursion
# Author: Jony Karki
# Date: Oct 14, 2017

# ask the user for the input and print the value
def main():
    n = eval(input("Enter a Non-Negative Number: "))
    print("{} {}".format("The Factorial of the number is", factorial(n)))

# compute the factorial of the passed number
def factorial(n):
    if n == 0:      # base case
        return 1
    else:
        return n * factorial(n - 1)

# run the main method
if __name__ == "__main__":
    main()