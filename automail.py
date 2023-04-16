#find documentation for ezgmail module at https://pypi.org/project/EZGmail/
#simple simon says module that interacts with google API to read the subject line of an email and respond to "Simon says:"
#DO NOT FORGET TO ADD CREDENTIALS.JSON AND TOKEN.JSON TO .GITIGNORE!!!

import ezgmail, re, time

check = True
while check:
    recThreads = ezgmail.recent()
    findEmail = re.compile(r'<(.*)@(.*)>')
    i = 0
    for msg in recThreads:
        subEval = recThreads[i].messages[0].subject.split(' ')
        sender = recThreads[i].messages[0].sender
        if subEval[0] == 'Simon' and subEval[1] == 'says:':
            subEval.remove('Simon')
            subEval.remove('says:')
            replyAddress = findEmail.search(sender).group(0).replace('<','').replace('>','')
            replyContent = 'I am now doing ' + ' '.join(subEval)
            ezgmail.send(replyAddress, replyContent, replyContent)
            ezgmail._trash(recThreads[i])
        if subEval[0] == 'ENDTASK': #remote kill command
            check = False
        i += 1
    time.sleep(60) #change check frquency; default every minute