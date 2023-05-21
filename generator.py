#!/usr/bin/env python3
import sys
import sympy as sp
import numpy as np
a,b ,c, d,e = sp.symbols("a, b, c, d,e")



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
    return retexercise,retsolution

def getfullname(exercise):
    match exercise:
        case "sole":
            return "System of linear equations"
    return exercise
    

def getExercise(exercise, Difficulty, Number):
    retsolution = ""
    retexercise = ""
    if len(exercise) ==0:
        return retexercise, retsolution
    name = getfullname(exercise) + " Difficulty: " + str(Difficulty)
    retexercise += "\\subsection{"+name+"} \n"
    retsolution += "\\subsection{"+name+"} \n"

    for i in range(Number):
        retexercise += "\\subsubsection{" "} \n"
        retsolution += "\\subsubsection{" "} \n"
        ex,sol =choseexercise(exercise,Difficulty)
        retexercise +=ex
        retsolution +=sol
    return retexercise, retsolution


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
        if argument ==1:
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
         
