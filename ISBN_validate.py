def validate_isbn(isbn):
    if len(isbn) == 10:
        # ten
        sum = 0
        for i in range(len(isbn)):
            sum += int(isbn[i]) * (10-i)
        
        if sum % 11 == 0:
            print("Valid ISBN!")
        else:
            print("Invalid ISBN!")
    elif len(isbn) == 13:
        sum = 0

        for i in range(len(isbn)):
            if i+4 % 2 == 0:
                sum += int(isbn[i])
            else:
                sum += int(isbn[i]) * 3

        if sum % 10 == 0:
            print("Valid ISBN!")
        else:
            print("Invalid ISBN")


if __name__ == "__main__":
    isbn = input("Enter the ISBN number: ")
    validate_isbn(isbn)
