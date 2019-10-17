'''
Author- DIWAKAR JAISWAL
find lenth Longest increasing subsequence of given array.
'''


def lis(a):
	n=len(a)
	#initialize ans array same lenth as 1
	ans=[1]*n
	for i in range(1,n):
		#now compare with first index to that index
		for j in range(i):
			if a[i]>a[j] and ans[i]<ans[j]+1:
				ans[i]=ans[j]+1
	return max(ans)

a=[1,3,2,6,4]

#longest increasing subsequence=[{1<3<6},{1<3<4},{1<2<6},{1<2<4}] length is 3

print("Maximum Length of longest increasing subsequence ",lis(a))
