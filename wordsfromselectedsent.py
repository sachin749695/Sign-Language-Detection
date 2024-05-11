kk = set()
with open("sentencesselected.txt","r") as f:
 	for i in f:
 		print(i.split("\t")[1].replace("\n",""))
 		k = i.split("\t")[1].replace("\n","")
 		# print(k.split())
 		for j in k.split():
 			kk.add(j)

with open("wordsselected.txt","a+") as f:
 	for i in kk:
 		f.write(i+'\n')

print(kk)
print(len(kk))


##sentences = []
##with open("sentencesselected.txt","r") as f:
##	for i in f:
##		print(i.split("\t")[0].replace("\n",""))
##		k = i.split("\t")[0].replace("\n","")
##		sentences.append(k)
##print(sentences)
