import imaplib
import email
from email.header import decode_header
import csv
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import svm
from sklearn.model_selection import GridSearchCV

username = input("Enter your mail id : ")
password = input("Password : ")

imap = imaplib.IMAP4_SSL("imap.gmail.com")
result = imap.login(username, password)
imap.select('"[Gmail]/All Mail"', readonly=True)

response, messages = imap.search(None, 'ALL')
messages = messages[0].split()
latest = int(messages[-1])
oldest = int(messages[0])

for i in range(latest, latest-1, -1):
	res, msg = imap.fetch(str(i), "(RFC822)")
	for response in msg:
		if isinstance(response, tuple):
			msg = email.message_from_bytes(response[1])
			print("***************************")
			#print("\n From ,{}".format(msg["From"]))
			from_text=msg["From"]
			subject_text=msg["Subject"]
			#print("\n Subject ,{}".format(msg["Subject"]))
	for part in msg.walk():
		if part.get_content_type() == "text/plain":
			body = part.get_payload(decode=True)
			#print('Body: {}'.format(str(body.decode())))

text = body.decode()
#with open('mailbody.csv', 'a+', newline='\n') as file:
#	writer = csv.writer(file)
#	writer.writerow(text.join)
#	writer.writerow(body.decode())
	
with open('mailbody.csv', mode='w') as csv_file:
    fieldnames = ['Label', 'EmailText']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerow({'Label': 'ham', 'EmailText': text})

dataframe = pd.read_csv("spam.csv")
#print(dataframe.describe())

test_mail = pd.read_csv("mailbody.csv")
#print(test_mail.describe())

##Step2: Split in to Training and Test Data

x = dataframe["EmailText"]
y = dataframe["Label"]

x1 = test_mail["EmailText"]
y1 = test_mail["Label"]

x_train,y_train = x[0:4425],y[0:4425]
x_test,y_test = x1[:],y1[:]

#print(x_test.shape)

##Step3: Extract Features
cv = CountVectorizer()  
features = cv.fit_transform(x_train)

##Step4: Build a model
tuned_parameters = {'kernel': ['rbf','linear'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 100, 100, 1000]}

model = GridSearchCV(svm.SVC(), tuned_parameters)

model.fit(features,y_train)

#print(model.best_params_)
#Step5: Test Accuracy
mscore=model.score(cv.transform(x_test),y_test)
#print(mscore)

if mscore < 1:
	opt=input("you have recieved spam message\nDo you want to read the mail..\n\tIf yes press 1, else 0..\nEnter your choice : ")
	if(opt == "1"):
		print("\n From: {}".format(msg["From"]))
		print("Subject: {}".format(msg["Subject"]))
		print('Body: {}'.format(str(body.decode())))
	else:
		print('Program ended...')
else:
	print('you have received a ham mail\n')
	print("\n From: {}".format(msg["From"]))
	print("\n Subject: {}".format(msg["Subject"]))
	print('\nBody: {}'.format(str(body.decode())))