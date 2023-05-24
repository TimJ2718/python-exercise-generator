#!/usr/bin/env python3
from numbers import Rational
import sys
import sympy as sp
import numpy as np
import random
a,b ,c, d,e = sp.symbols("a, b, c, d,e")



def getrandomint(n):
    ret = 0
    while ret ==0:
        ret = np.random.randint(-n,n)
    return ret

def getunimodularmatrix(n):
    upmatrix = np.zeros((n,n), dtype=np.int8)
    downmatrix = np.zeros((n,n), dtype = np.int8)
    for i in range(n): 
        upmatrix[i][i]=1
        downmatrix[i][i]=1
        for j in range(i): 
            upmatrix[i][j]=  getrandomint(4)
            downmatrix[j][i] = getrandomint(4)
    upmatrix = sp.Matrix(upmatrix)
    downmatrix = sp.Matrix(downmatrix)
    retmatrix =upmatrix * downmatrix
   
    return retmatrix

def printnpvec(vector):
    return sp.latex(sp.Matrix(vector))

def getdiagonalmatrix(n):
    matrix = np.zeros((n,n),dtype = np.int8)
    if n <=2:
        for i in range(n):
            matrix[i][i] = getrandomint(7)
        return matrix
    for i in range(n-2):
        matrix[i][i]=np.random.randint(-1,1)
    for i in range(n-1,n):        
        matrix[i][i]= getrandomint(7)
    return matrix

def getvarvector(n):
    match n:
        case 2:
            varvector = sp.Matrix([[a], [b]])
        case 3:
            varvector = sp.Matrix([[a], [b], [c]])
        case 4:
            varvector = sp.Matrix([[a], [b], [c],[d]])
        case 5:
            varvector = Sp.Matrix([[a],[b],[c],[d]])
    return varvector

def getmatrix(n,m):
    matrix = np.random.randint(-10, 10, size=(n,m))
    return sp.Matrix(matrix)
def geteasymatrix(n):
    while True:
        matrix = np.random.randint(-10, 10, size=(n,n))
        matrix = sp.Matrix(matrix)
        if abs(matrix.det()) < 20:
            return matrix

def getcolumn(n):
    matrix = np.random.randint(-10,10, size=(n,1))
    return sp.Matrix(matrix)

def getrow(n):
    matrix = np.random.randint(-10,10, size=(1,n))
    return sp.Matrix(matrix)


def Sole(Difficulty): #System of linear equations
    
    solution = ""
    exercise = ""
    n = Difficulty +1
    
    varvector = getvarvector(n)
    matrix = getmatrix(n,n)
    vector = getmatrix(n,1)

    leftside = matrix * varvector
    rightside = matrix *vector

    exercise = "$" +sp.latex(leftside) + " = " + sp.latex(rightside) +"$ \n"
    solution = "$" + sp.latex(varvector) + "=" + sp.latex(vector) + "$ \n"
    return exercise, solution

def eigenvalues(Difficulty): #Characteristical polynom, eigenvalues, eigenvectors
    solution = ""
    exercise = ""
    n = Difficulty +1
    while True:
        Tmatrix = getunimodularmatrix(n)
        invTmatrix= Tmatrix.inv()
        diagonal = getdiagonalmatrix(n)
        A = Tmatrix*diagonal*invTmatrix
        if np.max(np.abs(A))<30:
            break
    exercise = "$"+ sp.latex(A) + "$"
    lamda = sp.symbols('lamda')
    p = A.charpoly(lamda).as_expr()
    solution += "$ \\chi(\\lambda )= " + sp.latex(p) +"$ \\\ \n"
    lam= A.eigenvals()
    solution += "$\lambda = "+ sp.latex(lam) +" $ \\\ \n"
    eigenvecs = A.eigenvects()
    solution += "$"+sp.latex(eigenvecs)+"$ \\\ \n"

    return exercise, solution


def recursion(): #Recursion equation => Make closed equation
    solution = ""
    exercise = ""
    while True:
        matrix = np.zeros((2,2), dtype=np.int8)
        eval1 = getrandomint(4)
        eval2 = getrandomint(4)
        #Ensure eigenvalues are ganzzahlig
        p = -(eval1 +eval2)
        q = eval1* eval2
        matrix[0][0]= -p
        matrix[0][1]= -q
        matrix[1][0]=1
        
        if sp.Matrix(matrix).is_diagonalizable() and p !=0:
            break
    if np.random.randint(2):
        exercise += "$a_n = " + str(matrix[0][0])  + " a_{n-1} +" +str(matrix[0][1]) + "a_{n-2} $ \\\ \n"
    else:
        exercise += "$a_{n+1} = " + str(matrix[0][0])  + "a_{n} +" +str(matrix[0][1]) + "a_{n-1} $ \\\ \n"
    matrix = sp.Matrix(matrix)
    case = np.random.randint(2)
    a0 = getrandomint(4)
    a1 = getrandomint(4)
    if case:
        exercise += "$a_0 ="+str(a0)+"; \\;\\;\\;\\;\\;  a_1 = "+str(a1) +"$ \\\ \n"
        vecr = "\\left[\\begin{matrix}a_{1} \\\ a_{0} \end{matrix}\\right]"
        vecl = "\\left[\\begin{matrix}a_{2} \\\ a_{1} \end{matrix}\\right]"
        m = 1
        nm = sp.symbols("n-1")

    else:
        exercise += "$a_1 ="+str(a0)+"; \\;\\;\\;\\;\\;  a_2 = "+str(a1) + "$ \\\ \n"
        vecr = "\\left[\\begin{matrix}a_{2} \\\ a_{1} \end{matrix}\\right]"
        vecl = "\\left[\\begin{matrix}a_{3} \\\ a_{2}\end{matrix}\\right]"
        m = 2
        nm = sp.symbols("n-2")
    #vec1 = sp.Matrix(["a_n","a_n-1"])
    solution += "Set up the matrix M: $" + vecl + "=" + sp.latex(matrix) + "\\cdot"  + vecr +"$ \\\ \n"
    vecl = "\\left[\\begin{matrix}a_{n} \\\ a_{n-1} \end{matrix}\\right]"
    solution += "General relation: $" + vecl + "=" + sp.latex(matrix) + "^{n-"+str(m)+"} \\cdot"  + vecr +"$ \\\ \n"
    solution += "Determine eigenvalues and eigenvectors: \\\ \n"
    lam= matrix.eigenvals()
    solution += "$\lambda = "+ sp.latex(lam) +" $ \\\ \n"
    eigenvecs = matrix.eigenvects()
    solution += "$"+sp.latex(eigenvecs)+"$ \\\ \n"
    (T,D) = matrix.diagonalize()
    Tinv = T.inv()
    solution += "Determine Diagonal Matrix M and Transforation Matrix T: $M=T^{-1}MT:$ \\\ \n"
    solution += "$D= " + sp.latex(D) + "; \\;\\;\\;\\;  T="+ sp.latex(T) +"   => T^{-1}= "+sp.latex(Tinv) + "$ \\\ \n" 
    solution += "Calculate $M^{n-"+str(m)+"} = (TDT^{-1})^{n-"+str(m)+"} = TD^{n-"+str(m)+"}T^{-1}$ \\\ \n"  
    
    Dpot = D**(nm) 
    n = sp.symbols("n")
    Dpot2 = D**(n)*D**(-m)
    MPoT = T*Dpot2*Tinv
    solution += "$M^{n-"+str(m)+"} =" + sp.latex(T) + sp.latex(Dpot) + sp.latex(Tinv) +"="+sp.latex(MPoT)+"$ \\\ \n"  
    vector = sp.Matrix([[a1],[a0]])
    vectorsolution = MPoT*vector
    solution += "Calculate $a_n$: \\\ \n"
    solution += "$"+vecl +"=" + sp.latex(MPoT) + sp.latex(vector) +"=" + sp.latex(vectorsolution)+"$ \\\ \n"
    return exercise,solution

def gramschmidt(Difficulty):
    retexercise =""
    retsolution =""
    n = Difficulty+2
    num = Difficulty +1
    genum = int(n/2)
    index_list=np.arange(0,n,1).tolist()
    maxnum = 4
    vectors1 = []
    for i in range(genum):
        vec1 = np.zeros((n,1), dtype=np.int32)
        vec2 = np.zeros((n,1), dtype=np.int32)
        i1=index_list.pop(random.randrange(len(index_list)))
        i2=index_list.pop(random.randrange(len(index_list)))
        vec1[i1] = getrandomint(maxnum)
        vec1[i2] = vec1[i1]
        while vec1[i2] ==vec1[i1]:
            vec1[i2] = getrandomint(maxnum)
        r = [-3,-2,2,3]
        factor = random.choice(r)
        vec2[i1]=vec1[i2]*factor
        vec2[i2]=-vec1[i1]*factor
        vectors1.append(vec1)
        vectors1.append(vec2)

    vectors2 =[]
    for i in range(num):
        vec=vectors1.pop(random.randrange(len(vectors1)))
        vectors2.append(vec)
    vectors = []
    for i in range(num):
        vec = vectors2[i].copy()
        r = [-4,-3,-2,2,3,4]
        factor = random.choice(r)
        for j in range(i):
            vec +=factor*vectors2[j].copy()
        vectors.append(vec)
    vectors = np.array(vectors)
    retexercise += "Basis: $B= \\left\\{"
    for i in range(len(vectors)):
        retexercise+=printnpvec(vectors[i])
        retexercise+=","
    #retexercise = retexercise[:-1]
    retexercise += "\\right\\}$ \n"

    newvectors = []
    for i in range(num):
        retsolution +="$v_{"+str(i+1)+"}=w_{"+str(i+1)+"}"
        for j in range(i):
            retsolution += "-\\frac{<v_"+str(j)+ ",w_{"+str(i+1)+"}>}{<v_{"+str(j+1)+"}v_{"+str(j+1)+"} >} v_{"+str(j+1)+"}"
        retsolution += "="+printnpvec(vectors[i])
        newvec = vectors[i]
        for j in range(i):
            vsquared = np.sum(newvectors[j]**2)
            skalar = np.sum(newvectors[j]*vectors[i])
            factor = int(skalar/vsquared)
            newvec =newvec - factor*newvectors[j].copy()
            retsolution+="- \\frac{"+str(skalar)+"}{"+str(vsquared)+"}"+printnpvec(newvectors[j])
        newvectors.append(newvec)
        if i >0:
            retsolution += "=" + printnpvec(newvectors[i])
        retsolution +="$\\\ \n "
    return retexercise,retsolution

def inverse(difficulty):
    retexercise =""
    retsolution =""
    n = difficulty+1
    r = [-2,2]
    factor = random.choice(r)
    matrix = factor*getunimodularmatrix(n)
    retexercise+="$"+ sp.latex(matrix) + "$ \n"
    retsolution+="$"+ sp.latex(matrix.inv())+"$ \n"
    return retexercise, retsolution


def detcase1():
    retexercise =""
    retsolution =""
    matrix = geteasymatrix(3)
    ri1 = random.randint(0,2) #copied row
    ri2 = random.randint(ri1+1,3) #place of row insertion
    ci1 = random.randint(0,3) #place of row colum sertion
    r = [-2,-1,1,2]
    factor = random.choice(r)
    row = matrix.row(ri1) * factor
    col = getcolumn(4)
    matrix2 = matrix.row_insert(ri2,row)
    matrix2 = matrix2.col_insert(ci1,col)
    detfactor = matrix2[ri2,ci1]-factor * matrix2[ri1,ci1]

    matrix3 = matrix2.copy()
    matrix3.row_del(ri2)

    row = np.zeros((1,4), dtype = np.int32)
    row[0,ci1] = detfactor
    row = sp.Matrix(row)
    matrix3 = matrix3.row_insert(ri2,row)
    if random.randint(0,1):
        retexercise+="$"+sp.latex(matrix2) + "$ \n"
        retsolution +="Add "+str(-factor)+" times row " + str(ri1+1) +" to row " +str(ri2+1) +": \\\ \n"
        retsolution += "$\\det \\left("+sp.latex(matrix2) +"\\right) = \\det \\left(" +sp.latex(matrix3) + "\\right) \\\ \n"
        retsolution += "=" + str(detfactor) + "\\cdot (-1)^{"+str(ri2+1)+"+"+str(ci1+1)+"} \\cdot \\det \\left(" + sp.latex(matrix) + " \\right)="
        retsolution += sp.latex(matrix2.det()) +"$ \n"
    else:
        matrix = matrix.T
        matrix2 = matrix2.T
        matrix3 = matrix3.T
        retexercise+="$"+sp.latex(matrix2) + "$ \n"
        retsolution +="Add "+str(-factor)+" times column " + str(ri1+1) +" to column " +str(ri2+1) +": \\\ \n"
        retsolution += "$\\det \\left("+sp.latex(matrix2) +"\\right) = \\det \\left(" +sp.latex(matrix3) + "\\right) \\\ \n"
        retsolution += "=" + str(detfactor) + "\\cdot (-1)^{"+str(ri2+1)+"+"+str(ci1+1)+"} \\cdot \\det \\left(" + sp.latex(matrix) + " \\right)="
        retsolution += sp.latex(matrix2.det()) +"$ \n"
    return retexercise, retsolution

def detcase2():
    retexercise = ""
    retsolution = ""
    while True:
        matrix = getmatrix(4,4)
        matrix[0,2]=0
        matrix[0,3]=0
        matrix[1,2]=0
        matrix[1,3]=0
        if abs(matrix.det())<100 and matrix.det() !=0:
            break



    if random.randint(0,1):
        matrix=matrix.T
        ri1 = random.randint(2,3) #row which is changed
        ri2 = random.randint(0,1) #row which is copied for change
    else:
        ri1 = random.randint(0,1)
        ri2 = random.randint(2,3)

    matrix2 = matrix.copy()
    r = [-2,-1,1,2]
    factor = random.choice(r)
    row=matrix2.row(ri1)+matrix2.row(ri2)*factor
    matrix2.row_del(ri1)
    matrix2 = matrix2.row_insert(ri1,row)
    m1 = sp.Matrix([[matrix[0,0],matrix[0,1]],[matrix[1,0],matrix[1,1]]])
    m2 = sp.Matrix([[matrix[2,2],matrix[2,3]],[matrix[3,2],matrix[3,3]]])

    if random.randint(0,1):
        retexercise+="$"+sp.latex(matrix2)+ "$ \n"
        retsolution += "Add "+str(-factor)+ " times row " + str(ri2+1) + " to row " + str(ri1+1) +": \\\ \n"
        retsolution += "$\\det \\left("+sp.latex(matrix2)+"\\right) = \\det \\left("+sp.latex(matrix) + "\\right) \\\ \n"
        retsolution += "\\det \\left("+sp.latex(m1)+" \\right) \cdot \\det \\left("+sp.latex(m2)+" \\right) ="
        retsolution +=sp.latex(m1.det())+"\\cdot("+sp.latex(m2.det())+")="+sp.latex(matrix2.det())+"$ \n"
    else:
        matrix2 = matrix2.T
        matrix = matrix.T
        m1 = m1.T
        m2 = m2.T
        retexercise+="$"+sp.latex(matrix2)+ "$ \n"
        retsolution += "Add "+str(-factor)+ " times column " + str(ri2+1) + " to column " + str(ri1+1) +": \\\ \n"
        retsolution += "$\\det \\left("+sp.latex(matrix2)+"\\right) = \\det \\left("+sp.latex(matrix) + "\\right) \\\ \n"
        retsolution += "\\det \\left("+sp.latex(m1)+" \\right) \cdot \\det \\left("+sp.latex(m2)+" \\right) ="
        retsolution +=sp.latex(m1.det())+"\\cdot("+sp.latex(m2.det())+")="+sp.latex(matrix2.det())+"$ \n"
    return retexercise, retsolution

def detcase3():
    retexercise = ""
    retsolution =""
    matrix = getmatrix(4,4)
    row_list = [0,1,2,3]
    i1 = row_list.pop(random.randrange(len(row_list)))
    i2 = row_list.pop(random.randrange(len(row_list)))
    i3 = row_list.pop(random.randrange(len(row_list)))
    r = [-1,1]
    factor1 = random.choice(r)
    factor2 = random.choice(r)
    row=matrix.row(i1)*factor1+matrix.row(i2)*factor2
    matrix.row_del(i3)
    matrix = matrix.row_insert(i3,row)

    retexercise+="$"+sp.latex(matrix)+"$ \n"
    retsolution += "Row " +str(i3+1) +" is given by $"+str(factor1)+"\\cdot$ row " +str(i1+1) +"+ $(" +str(factor2)+ ")\\cdot$ row " +str(i2+1) +"\\\ \n"
    retsolution +="$\\det \\left(" +sp.latex(matrix) + "\\right) = 0 \n$"

    return retexercise, retsolution


def determinant(difficulty):
    retexercise =""
    retsolution = ""
    n = difficulty+1
    if n < 4:
        matrix = geteasymatrix(n)
        retexercise+="$"+ sp.latex(matrix) +"$ \n"
        retsolution+="$\\det(M)="+sp.latex(matrix.det())+"$"
        return retexercise, retsolution
    match random.randint(0,2):
        case 0:
            return detcase1()
        case 1:
            return detcase2()
        case 2:
            return detcase3()



    return retexercise, retsolution

def choseexercise(exercise,Difficulty):
    retexercise = ""
    retsolution =""
    if Difficulty < 1:
        Difficulty =1
    elif Difficulty > 3:
        Difficulty = 3
    match exercise:
        case "sole":
            retexercise, retsolution = Sole(Difficulty)
        case "eval":
            retexercise, retsolution = eigenvalues(Difficulty)
        case "recursive":
            retexercise, retsolution = recursion()
        case "gramschmidt":
            retexercise, retsolution = gramschmidt(Difficulty)
        case "inverse":
            retexercise, retsolution = inverse(Difficulty)
        case "determinant":
            retexercise, retsolution = determinant(Difficulty)

    return retexercise,retsolution

def getfullname(exercise):
    retname = exercise
    retdescription =""
    match exercise:
        case "sole":
            retname = "System of linear equations"
            retdescription = "Solve the linear equations:"
        case "eval":
            retname = "Eigenvalue problem"
            retdescription = "Determine the characteristical polynom, the eigenvalues and the eigenvectors:"
        case "recursive":
            retname = "Recursive equation"
            retdescription = "Use methods of the linear algebra to determine a closed form of the recursion equation:"
        case "gramschmidt":
            retname = "Gram schmidt"
            retdescription = "Use the gram schmidt method to orthogonalize the given basis:"
        case "inverse":
            retname = "Inverse"
            retdescription = "Calculate the inverse of the given matrix:"
        case "determinant":
            retname = "Determinant"
            retdescription = "Calculate the determinant of the given matrix:"
    return retname, retdescription
    

def getExercise(exercise, Difficulty, Number):
    retsolution = ""
    retexercise = ""
    if len(exercise) ==0:
        return retexercise, retsolution
    name, Description = getfullname(exercise) 
    name += "; Difficulty: " + str(Difficulty)
    retexercise += "\\subsection{"+name+"}  \n"
    retexercise += Description + " \n"
    retsolution += "\\subsection{"+name+"} \n"

    for i in range(Number):
        retexercise += "\\subsubsection{" "}  \n"
        retsolution += "\\subsubsection{" "} \n"
        ex,sol =choseexercise(exercise,Difficulty)
        retexercise +=ex
        retsolution +=sol
    return retexercise, retsolution


def printhelp():
    printtext = "For genearating a exerecise use the command \"python generator.py exercisename \" \n"
    printtext += "You can use -n to specify the amount of exercises and -d to select the difficulty from 1,2,3 \n"
    printtext += "The default difficulty is 1, the default amount of exercises is also 1 \n"
    printtext += "You can generate multiple exercises at once with \"python generator.py name1 name2 -d 3 name3 -n 2 \" \n \n"
    printtext += "List of exercises: \n"
    printtext += "System of linear equations: sole \n"
    printtext += "Characteristical polynom,Eigenvalues, Eigenvector: eval \n"
    printtext += "Recursion equation: recursion \n"
    printtext += "Gram schmidt: gramschmidt \n"
    printtext += "Determinant: determinant \n"
    print(printtext)

if __name__ == "__main__":
    exercisetex= ""
    solutiontex =""
    outputtex =""

    exercise = ""
    Number = 1
    Difficulty = 1
    argument = 0
    for i in range(1,len(sys.argv)):
        zw = sys.argv[i]
        if zw =="-h" or zw =="--h" or zw =="-help" or zw=="--help":
          printhelp()
        elif argument ==1:
            Number = int(zw)
            argument =0
        elif argument ==2:
            Difficulty = int(zw)
            argument = 0
        elif zw == "-n":
            argument = 1
        elif zw == "-d":
            argument = 2
        else:
            ex,sol=getExercise(exercise,Difficulty,Number)
            exercisetex += ex
            solutiontex += sol
            Number = 1
            Difficulty = 1
            exercise = zw

    ex,sol=getExercise(exercise,Difficulty,Number)
    exercisetex += ex
    solutiontex += sol

    outputtex  = "\documentclass[10pt,a4paper]{article} \n"
    outputtex +="\\usepackage[utf8]{inputenc} \n"
    outputtex +="\\usepackage{amsmath} \n"
    outputtex +="\\usepackage{amsfonts} \n"
    outputtex +="\\usepackage{amssymb} \n"
    outputtex +="\\begin{document} \n"
    outputtex += "\\section{Exercises} \n"
    outputtex += exercisetex 
    outputtex += "\\clearpage \n"
    outputtex +="\\section{Solutions} \n"
    outputtex += solutiontex
    outputtex +="\\end{document} \n"
    text_file = open("Exercise.tex", "w")
    text_file.write(outputtex)
    text_file.close()        
         
