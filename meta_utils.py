#   CS 292, Fall 2009
#   Final Project
#   Dwight Bell
#--------------------

"""
Code-writing utilities for generating code within Source Modules.
These routines generate sections of code as a string.

    loop(start, end, factor, string) - generates an unrolled loop
    
    copy_to_shared(strType, strGlobal, strShared, size) - copies an
        array from global memory to shared memory.
    
    reduction2(strData1, strData2, blocksize) - performs a
        reduction sum on on two blocks of data.  The blocks must
        have size equal to a power of two, less thn or equal to 512.    
"""


def loop(start, end, factor, string):
    """
    loop(start, end, factor, string)
    
    Generates an unrolled for loop.  The string argument contains the code
    within the for loop.  The code in string can use the loop index by
    referring to {0}.
        start   start value for loop variable __i__
        end     end value for loop.  The test is __i__ < end
        factor  unrolling factor = the number of iterations of the loop code
                within the generated loop
        string  code for inside the loop, using {0} as references to loop
                variable
    """
    reps = (end-start)/factor
    code = ""
    if reps > 0:
        code += "for(int __i__={0}; __i__<{1}; __i__+= {2})".format(start, 
                                                                start + factor*reps - 1, factor)
        code += "{\n"
        for j in range(factor):
            code += "   "+string.format("(__i__+{0})".format(j))
        code += "\n}\n"
    for k in range(start + factor*reps, end):
        code += string.format(k)
    return code

    

#maximum amount of shared memory
MU_MAX_SHARED = 16384/4 - 64       # assumes elements are 32-bit


def copy_to_shared(strType, strGlobal, strShared, size):
    """
    copy_to_shared(strType, strGlobal, strShared, size)
    
    Generates a code segment to copy from global to shared memory.  Will use 
    global memory instead if the size of the data is greater than MAX_SHARED.
    Assumes elements are 4 bytes.
    Assumes threads are synchronized on entry.
        strType    data type of the global and shared memory
        strGlobal  global data, as a string,
        strShared  name for shared dataas a string
        size       number of elements in the data (assumed to be 4 bytes each)
    """
    if(size < MU_MAX_SHARED):
        return """
    __shared__ """ + strType + " " + strShared + "[" + str(size) + """];
    for(int __idx__ = threadIdx.x; __idx__ < """ + str(size) + """; __idx__ += blockDim.x){
        """ + strShared + """[__idx__] = """ + strGlobal + """[__idx__];
    }
    __syncthreads();
    """
    else:
        return strType + "* " + strShared + " = " + strGlobal + ";\n"


def copy_to_shared_32(strType, strGlobal, strShared, size):
    """
    copy_to_shared_32(strType, strGlobal, strShared, size)
    
    Same format as copy_to_shared, but uses the 32 threads in a warp to
    do all the copying.  No need to synchronize when done.
    """
    if(size < MU_MAX_SHARED):
        return """
    __shared__ """ + strType + " " + strShared + "[" + str(size) + """];
    for(int __idx__ = threadIdx.x % 32; __idx__ < """ + str(size) + """; __idx__ += 32){
        """ + strShared + """[__idx__] = """ + strGlobal + """[__idx__];
    }
    """
    else:
        return strType + "* " + strShared + " = " + strGlobal + ";\n"


def reduction2(strData1, strData2, blocksize):
    """
    reduction2(strData1, strData2)
    
    Performs a reduction sum on two sets of data.
    """
    
    code = """
        __syncthreads();
        unsigned int __idx__ = threadIdx.x;
        """

    if blocksize >= 512:
        code += """
        if (__idx__ < 256) { 
            """ + strData1 + "[__idx__] += " + strData1 + """[__idx__ + 256];
            """ + strData2 + "[__idx__] += " + strData2 + """[__idx__ + 256];
        }
        __syncthreads();
        """

    if blocksize >= 256:
        code += """
        if (__idx__ < 128) { 
            """ + strData1 + "[__idx__] += " + strData1 + """[__idx__ + 128];
            """ + strData2 + "[__idx__] += " + strData2 + """[__idx__ + 128];
        } 
        __syncthreads(); 
        """
    
    if blocksize >= 128:
        code += """
        if (__idx__ < 64) { 
            """ + strData1 + "[__idx__] += " + strData1 + """[__idx__ + 64];
            """ + strData2 + "[__idx__] += " + strData2 + """[__idx__ + 64];
        } 
        __syncthreads(); 
        """

    code += """
        if (__idx__ < 32){
        """
    if blocksize >= 64:
        code += """
                """ + strData1 + "[__idx__] += " + strData1 + """[__idx__ + 32];
                """ + strData2 + "[__idx__] += " + strData2 + """[__idx__ + 32];
                """
    if blocksize >= 32:
        code += """
                """ + strData1 + "[__idx__] += " + strData1 + """[__idx__ + 16];
                """ + strData2 + "[__idx__] += " + strData2 + """[__idx__ + 16];
                """
    if blocksize >= 16:
        code += """
                """ + strData1 + "[__idx__] += " + strData1 + """[__idx__ + 8];
                """ + strData2 + "[__idx__] += " + strData2 + """[__idx__ + 8];
                """
    if blocksize >= 8:
        code += """
                """ + strData1 + "[__idx__] += " + strData1 + """[__idx__ + 4];
                """ + strData2 + "[__idx__] += " + strData2 + """[__idx__ + 4];
                """
    if blocksize >= 4:
        code += """
                """ + strData1 + "[__idx__] += " + strData1 + """[__idx__ + 2];
                """ + strData2 + "[__idx__] += " + strData2 + """[__idx__ + 2];
                """
    if blocksize >= 2:
        code += """
                """ + strData1 + "[__idx__] += " + strData1 + """[__idx__ + 1];
                """ + strData2 + "[__idx__] += " + strData2 + """[__idx__ + 1];
                """
    code += """
        }
        """

    return code


