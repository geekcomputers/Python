import os


import  mysql.connector as mys
mycon=mys.connect(host='localhost',user='root',passwd='Yksrocks',database='book_store_management')


if mycon.is_connected():
    print()
    print('successfully connected')

mycur=mycon.cursor()








def  DBZ():


    # IF  NO.  OF  BOOKS  IS     ZERO(0)     THAN  DELETE  IT  AUTOMATICALLY



    display="select * from books"
    mycur.execute(display)
    data2=mycur.fetchall()


    for y in data2:
    
        if y[6]<=0:

            
            delete="delete from books where  Numbers_of_book<=0"
            mycur.execute(delete)
            mycon.commit()







            
def separator():
    print()
    print("\t\t========================================")
    print()




    
def end_separator():
    print()
    print()







    
def   login():

    
    user_name=input(" USER NAME  ---  ")
    passw=input(" PASSWORD  ---  ")


    display='select * from login'
    mycur.execute(display)
    data2=mycur.fetchall()


    for y in data2:

        if y[1]==user_name  and  y[2]==passw:

            pass

        else:
                    
            
            separator()

            
            print(" Username  or  Password  is  Incorrect  Try Again")


            separator()

                    
            user_name=input(" USER NAME  ---  ")
            passw=input(" PASSWORD  ---  ")


            if y[1]==user_name  and  y[2]==passw:
                        
                pass

            else:

                separator()
                        
                print(" Username  or  Password  is  Again  Incorrect")
                exit()







                
def ViewAll():
    
    print("\u0332".join("BOOK NAMES~~"))
    print("------------------------------------")

    
    display='select * from books'
    mycur.execute(display)
    data2=mycur.fetchall()
    c=0

    
    for y in data2:

        c=c+1
        print(c,"-->",y[1])







        
def  CNB1():


    if y[6]==0:

        separator()

        print(" NOW  THIS  BOOK  IS  NOT  AVAILABLE ")



    elif y[6]>0 and y[6]<=8:

        separator()

        print("WARNING!!!!!!!!!!!!!!!!!!!!!!!")
        print("NO.  OF THIS BOOK IS LOW","\tONLY",y[6]-1,"LEFT")


        print()
        print()
        
        
    
    elif y[6]>8:

        separator()
        
        print("NO.  OF  BOOKS  LEFT  IS ",y[6]-1)


        print()
        print()







def  CNB2():


    if y[6]<=8:

        separator()

        print("WARNING!!!!!!!!!!!!!!!!!!!!!!!")
        print("NO.  OF THIS BOOK IS LOW","\tONLY",y[6],"LEFT")


        

                                

    else:

        separator()

        print("NO.  OF  BOOKS  LEFT  IS ",y[6])


        







        
separator()








# LOGIN



display12='select * from visit'
mycur.execute(display12)
data2222=mycur.fetchall()
for m in data2222:

    if m[0]==0:

        
        c=m[0] 
        display11='select * from login'
        mycur.execute(display11)
        data222=mycur.fetchall()


        if c==0:
            
            if  c==0:
                

                print("\t\t\t\t REGESTER     ")
                print("\t\t\t\t----------------------------")


                print()
                print()

    
                user_name=input("ENTER  USER  NAME -- ")
                passw=input("ENTER  PASSWORD  limit 8-20  -- ")
                lenght=len(passw)

                
                if  lenght>=8  and  lenght<=20:

                    c=c+1
                    insert55=(c,user_name,passw)
                    insert22="insert into login values(%s,%s,%s)"
                    mycur.execute(insert22,insert55)
                    mycon.commit()


                    separator()

    
                    login()



                    
                else:
                    

                    if  lenght<8:

                        
                        separator()

            
                        print(" Password Is less than  8  Characters  Enter Again")

                        
                        separator()
                        
            
                        user_name2=input("ENTER  USER  NAME -- ")
                        passw2=input("ENTER  PASSWORD AGAIN (limit 8-20) -- ")
                        lenght1=len(passw2)
                        
        
                        if  lenght1>=8  and  lenght1<=20:

                            
                            c=c+1
                            insert555=(c,user_name2,passw2)
                            insert222="insert into login values(%s,%s,%s)"
                            mycur.execute(insert222,insert555)
                            mycon.commit()

                            
                            separator()



                        

    
                            login()

                            


                        elif  lenght>20:


                            separator()

            
                            print(" Password Is  Greater  than  20  Characters  Enter Again")


                            separator()

            
                            user_name=input("ENTER  USER  NAME -- ")
                            passw=input("ENTER  PASSWORD AGAIN (limit 8-20) -- ")
                            lenght=len(passw)

        
                            if  lenght>=8  and  lenght>=20:

                                
                                c=c+1
                                insert55=(c,user_name,passw)
                                insert22="insert into login values(%s,%s,%s)"
                                mycur.execute(insert22,insert55)
                                mycon.commit()


                                separator()

    
                                login()




        update33="update visit set visits=%s"%(c)
        mycur.execute(update33)
        mycon.commit()
        
        

    elif m[0]==1:

        if  m[0]==1:

            login()





            
separator()






DBZ()







# REPETITION


a=True


while a==True:





    # PROGRAM STARTED





    print("     *TO VIEW ALL ENTER 1")
    print("     *TO SEARCH and BUY BOOK ENTER 2")
    print("     *TO ADD BOOK ENTER 3")
    print("     *TO UPDATE ENTER 4")
    print("     *TO DELETE BOOK ENTER 5")
    print("     *TO CLOSE ENTER 6")


    print()


    choice=int(input("ENTER YOUR CHOICE -- "))


    separator()










    #VIEW



    if  choice==1:

        print()
        
        ViewAll()
        


        separator()



        rep=input("Do  You  Want  To  Restart  ??    yes / no  --  ").lower()


        if  rep=="yes":

            end_separator()

            separator()

            DBZ()

            continue

        else:

            end_separator()

            DBZ()
            
            os._exit(0)




        end_separator()

        









        
    #SEARCH / BUY



    if  choice==2:

        
        book_name=input("ENTER BOOK NAME ---- ")


        separator()

        
        display="select * from books where Name='%s'"%(book_name)
        mycur.execute(display)
        data2=mycur.fetchone()

        
        if data2!=None:

            
            print("BOOK IS AVAILABLE")









            
            #BUY OR NOT



            separator()


            print("\t*WANT TO BUY PRESS 1")
            print("\t*IF NOT PRESS 2")
            print()

            
            choice2=int(input("ENTER YOUR CHOICE -- "))

            
            if choice2==1:









                
                #BUY 1 OR MORE



                separator()


                print("\t*IF YOU WANT ONE BOOK PRESS 1")
                print("\t*IF YOU WANT MORE THAN ONE BOOK PRESS 2")
                print()

                
                choice3=int(input("ENTER YOUR CHOICE -- "))

                
                if choice3==1:


                    display='select * from books'
                    mycur.execute(display)
                    data2=mycur.fetchall()

                    
                    for y in data2:
                        
                        if y[1]==book_name:
                                                                         
                                if y[6]>0:


                                    separator()


                                    u="update books set Numbers_of_book=Numbers_of_book - 1 where name='%s';"%(book_name)
                                    mycur.execute(u)
                                    mycon.commit()


                                    print("BOOK WAS BOUGHT")


                                    separator()


                                    print("THANKS FOR COMING")


                                    CNB1()




                                    separator()



                                    rep=input("Do  You  Want  To  Restart  ??    yes / no  --  ").lower()


                                    if  rep=="yes":

                                        end_separator()

                                        separator()

                                        DBZ()

                                        continue

                                    else:

                                        end_separator()

                                        DBZ()
                                        
                                        os._exit(0)




                          
                if choice3==2:


                    separator()


                    wb=int(input("ENTER NO. OF BOOKS -- "))


                    separator()


                    display='select * from books'
                    mycur.execute(display)
                    data2=mycur.fetchall()

                    
                    for y in data2:

                        if y[1]==book_name:

                            if  wb>y[6]:
                                                                         
                                if y[6]>0:
                                    

                                     print("YOU CAN'T  BUT  THAT  MUCH  BOOKS")


                                     separator()


                                     print("BUT YOU CAN BUY",y[6],"BOOKS MAX")


                                     separator()

                                     
                                     choice44=input("DO YOU WANT TO BUY BOOK ?     Y/N -- ")


                                     separator()

                                     
                                     k=y[6]

                                     
                                     if choice44=="y" or choice44=="Y":

                                         
                                         u2="update books set numbers_of_book=numbers_of_book -%s where name='%s'"%(k,book_name)
                                         mycur.execute(u2)
                                         mycon.commit()


                                         print("BOOK WAS BOUGHT")


                                         separator()

                                         
                                         print("THANKS FOR COMING")

                                         
                                         separator()


                                         display='select * from books'
                                         mycur.execute(display)
                                         data2=mycur.fetchall()

                                         
                                         for y in data2:
                                             
                                             if y[1]==book_name:
                                                 
                                                 if y[6]<=8:

                                                     
                                                     print("WARNING!!!!!!!!!!!!!!!!!!!!!!!")
                                                     print("NO.  OF THIS BOOK IS LOW","\tONLY",y[6],"LEFT")


                                                     end_separator()

                                                     
                                                     break



                                         separator()


                                         rep=input("Do  You  Want  To  Restart  ??    yes / no  --  ").lower()


                                         if  rep=="yes":

                                             end_separator()

                                             separator()

                                             DBZ()

                                             continue

                                         else:

                                             end_separator()

                                             DBZ()
                                            
                                             os._exit(0)



                                     elif  choice44=="n" or choice44=="N":

                                         
                                         print("SORRY  FOR  INCONVENIENCE  WE  WILL  TRY  TO  FULLFILL  YOUR  REQUIREMENT  AS  SOON  AS  POSSIBLE")


                                         end_separator()

                                            




                                         separator()

                                    

                                         rep=input("Do  You  Want  To  Restart  ??    yes / no  --  ").lower()


                                         if  rep=="yes":

                                             separator()

                                             DBZ()

                                             continue

                                         else:

                                             end_separator()

                                             DBZ()

                                             os._exit(0)



                                         
                                elif y[6]==0:

                                    
                                    print("SORRY  NO  BOOK  LEFT  WE  WILL  TRY  TO  FULLFILL  YOUR  REQUIREMENT  AS  SOON  AS  POSSIBLE")


                                    end_separator()

                                    


                                    separator()

                                    

                                    rep=input("Do  You  Want  To  Restart  ??    yes / no  --  ").lower()


                                    if  rep=="yes":

                                        separator()

                                        DBZ()

                                        continue

                                    else:
                                        
                                        end_separator()

                                        DBZ()

                                        os._exit(0)

     

                            else:

                                
                                u2="update books set numbers_of_book=numbers_of_book -%s where name='%s'"%(wb,book_name)
                                mycur.execute(u2)
                                mycon.commit()


                                print("BOOK WAS BOUGHT")


                                separator()

                                
                                print("THANKS FOR COMING")

                            
                                display='select * from books'
                                mycur.execute(display)
                                data2=mycur.fetchall()

                                
                                for y in data2:
                                    
                                    if y[1]==book_name:
                                        
                                        CNB2()


                                        separator()

                                    

                                        rep=input("Do  You  Want  To  Restart  ??    yes / no  --  ").lower()


                                        if  rep=="yes":

                                            separator()

                                            DBZ()

                                            continue

                                        else:

                                            end_separator()

                                            DBZ()

                                            os._exit(0)
                                        


            else:


                separator()

                
                print("NO BOOK IS BOUGHT")


                end_separator()

                                    




                separator()

                                    

                rep=input("Do  You  Want  To  Restart  ??    yes / no  --  ").lower()


                if  rep=="yes":

                    separator()

                    DBZ()

                    continue

                else:

                    end_separator()

                    DBZ()

                    os._exit(0)



                
        else:


            separator()

                
            print("SORRY NO BOOK WITH THIS NAME EXIST / NAME IS INCORRECT")


            end_separator()

                                            



            separator()

                                    

            rep=input("Do  You  Want  To  Restart  ??    yes / no  --  ").lower()


            if  rep=="yes":

                separator()

                DBZ()

                continue

            else:

                end_separator()

                DBZ()

                os._exit(0)









            
    # ADDING BOOK



    if  choice==3:

        
        q10=int(input("ENTER NO. OF BOOKS TO ADD -- "))


        separator()

        
        for k in range(q10):

            
            SNo10=int(input("ENTER SNo OF BOOK -- "))
            name10=input("ENTER NAME OF BOOK --- ")
            author10=input("ENTER NAME OF AUTHOR -- ")
            year10=int(input("ENTER YEAR OF PUBLISHING -- "))
            ISBN10=input("ENTER ISBN OF BOOK -- ")
            price10=int(input("ENTER PRICE OF BOOK -- "))
            nob10=int(input("ENTER NO. OF BOOKS -- "))



            display10="select * from books where ISBN='%s'"%(ISBN10)
            mycur.execute(display10)
            data20=mycur.fetchone()

            

            if data20!=None:
                
                print("This  ISBN Already Exists")

                os._exit(0)

            else:

                    
                insert=(SNo10,name10,author10,year10,ISBN10,price10,nob10)
                insert20="insert into books values(%s,%s,%s,%s,%s,%s,%s)"
                mycur.execute(insert20,insert)
                mycon.commit()


                separator()

        
                print("BOOK IS ADDED")


                separator()


        

                            

        rep=input("Do  You  Want  To  Restart  ??    yes / no  --  ").lower()


        if  rep=="yes":

            separator()

            DBZ()

            continue

        
        else:

            end_separator()

            DBZ()

            os._exit(0)









            
    # UPDATING BOOK



    if  choice==4:


        choice4=input("ENTER ISBN OF BOOK -- ")

        
        separator()

        
        display="select * from books where ISBN='%s'"%(choice4)
        mycur.execute(display)
        data2=mycur.fetchone()

        
        if data2!=None:

            
            SNo1=int(input("ENTER NEW SNo OF BOOK -- "))
            name1=input("ENTER NEW NAME OF BOOK --- ")
            author1=input("ENTER NEW NAME OF AUTHOR -- ")
            year1=int(input("ENTER NEW YEAR OF PUBLISHING -- "))
            ISBN1=input("ENTER NEW ISBN OF BOOK -- ")
            price1=int(input("ENTER NEW PRICE OF BOOK -- "))
            nob=int(input("ENTER NEW NO. OF BOOKS -- "))
            insert=(SNo1,name1,author1,year1,ISBN1,price1,nob,choice4)
            update="update books set SNo=%s,Name=%s,Author=%s,Year=%s,ISBN=%s,Price=%s,numbers_of_book=%s where ISBN=%s"
            mycur.execute(update,insert)
            mycon.commit()

            
            separator()

            
            print("BOOK IS UPDATED")

            
            separator()

                                    

            rep=input("Do  You  Want  To  Restart  ??    yes / no  --  ").lower()


            if  rep=="yes":

                separator()

                DBZ()

                continue

            
            else:

                end_separator()

                DBZ()

                os._exit(0)

            
        else:
            
            print("SORRY NO BOOK WITH THIS ISBN IS EXIST  /  INCORRECT ISBN")


            print()
            print()



            separator()

                                    

            rep=input("Do  You  Want  To  Restart  ??    yes / no  --  ").lower()


            if  rep=="yes":

                separator()

                DBZ()

                continue

            
            else:

                end_separator()

                DBZ()

                os._exit(0)








            
    # DELETING A BOOK



    if choice==5:

        
        ISBN1=input("ENTER ISBN OF THAT BOOK THAT YOU WANT TO DELETE -- ")
        display="select * from books where ISBN='%s'"%(ISBN1)
        mycur.execute(display)
        data2=mycur.fetchone()

        
        if data2!=None:


            separator()

            
            choice5=input("ARE YOU SURE TO DELETE THIS BOOK ENTER Y/N -- ")

            
            if  choice5=='Y' or choice5=='y':


                separator()

                
                ISBN2=input("PLEASE ENTER ISBN AGAIN -- ")
                delete="delete from books where ISBN='%s'"%(ISBN2)
                mycur.execute(delete)
                mycon.commit()

                
                separator()


                print("BOOK IS DELETED")


                print()
                print()




                separator()

                                    

                rep=input("Do  You  Want  To  Restart  ??    yes / no  --  ").lower()


                if  rep=="yes":

                    separator()

                    DBZ()

                    continue

                
                else:

                    end_separator()

                    DBZ()

                    os._exit(0)
                


            else:

                
                separator()

                
                print("NO BOOK IS DELETED")


                print()
                print()


                separator()

                                    

                rep=input("Do  You  Want  To  Restart  ??    yes / no  --  ").lower()


                if  rep=="yes":

                    separator()

                    DBZ()

                    continue

                
                else:

                    end_separator()

                    DBZ()

                    os._exit(0)

                

        else:


            separator()

            
            print("SORRY NO BOOK WITH THIS ISBN AVAILABLE / ISBN IS INCORRECT")


            print()
            print()



            separator()

                                    

            rep=input("Do  You  Want  To  Restart  ??    yes / no  --  ").lower()


            if  rep=="yes":

                separator()

                DBZ()

                continue

            
            else:

                end_separator()

                DBZ()

                os._exit(0)









            
    # CLOSE

    if choice==6:

        exit()
        os._exit(0)









    
# IF  NO.  OF  BOOKS  IS     ZERO(  0  )     THAN  DELETE  IT  AUTOMATICALLY



display="select * from books"
mycur.execute(display)
data2=mycur.fetchall()


for y in data2:
    
    if y[6]<=0:

            
        delete="delete from books where  Numbers_of_book<=0"
        mycur.execute(delete)
        mycon.commit()
