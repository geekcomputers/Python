def generate(A,k):
    if k ==1:
        print(A)
        return
    else:
        for i in range(k):
            generate(A,k-1)
            if(i<k-1):
                if k%2 == 0:
                    A[i],A[k-1] = A[k-1],A[i]
                else:
                    A[0],A[k-1] = A[k-1],A[0]

A = [1,2,3,4] #test-case
x = len(A)
generate(A,x)
