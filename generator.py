#!/usr/bin/env python3
from numbers import Rational
import sys
import sympy as sp
import numpy as np
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
    n = Difficulty +2
    num = Difficulty +1
    vectors =  np.random.randint(-7, 7, size=(n,n))
    print(1)
    print(vectors)
    print(2)
    vecs = sp.GramSchmidt(vectors)
    print(vecs)
    


    return retexercise,retsolution


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
        case "recursion":
            retexercise, retsolution = recursion()
        case "gramschmidt":
            retexercise, retsolution = gramschmidt(Difficulty)

    return retexercise,retsolution

def getfullname(exercise):
    retname = exercise
    retdescription =""
    match exercise:
        case "sole":
            retname = "System of linear equations"
            retdescription = "Solve the linear equations."
        case "eval":
            retname = "Eigenvalue problem"
            retdescription = "Determine the characteristical polynom, the eigenvalues and the eigenvectors"    
        case "recursion":
            retname = "Recursion equation"
            retdescription = "Use methods of the linear algebra to determine a closed form of the recursion equation"
        case "gramschmidt":
            retname = "Gram schmidt"
            retdescription = "Use the gram schmidt method to orthogonalize the given basis"
    
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
    print("sole;  eval; recursion")

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
          printhelp
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
         
