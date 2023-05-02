import sys
import numpy as np

act_width = 4
reshp_factor = 8

def read_image(file_name):
    nbytes = 0
    image_bin = []
    value = 0
    with open(file_name, "rb") as f:
        byte = f.read(1)
        while (byte != b""):
            bits = "{:08b}".format(int(byte.hex(),16))
#            print(bits[4:],end="  ")
            image_bin.append(bits[4:])
            nbytes = nbytes + 1
            byte = f.read(1)
#            print(weights_bin)
#    print(nbytes)
#    print("\n")
    return image_bin

    
#Convert binary number to signed int
def bin_2_signed_int(val, bits):
#       compute the 2's complement of int value val
    if (val & (1 << (bits - 1))) != 0: # if sign bit is set e.g., 8bit: 128-255
        val = val - (1 << bits)        # compute negative value
    return val   



def reshape_image(image, file_name):
    reshp_value = 0
    index= 0
    outarray = np.zeros(((9*9*200)//reshp_factor), dtype=np.uint32)
    for i in range(9*9):
            for z in range(0,200,reshp_factor):
                n0 = image[((z+0)*9*9) + i]
                n1 = image[((z+1)*9*9) + i]
                n2 = image[((z+2)*9*9) + i]
                n3 = image[((z+3)*9*9) + i]
                n4 = image[((z+4)*9*9) + i]
                n5 = image[((z+5)*9*9) + i]
                n6 = image[((z+6)*9*9) + i]
                n7 = image[((z+7)*9*9) + i]
                if index == 25:
                    print("index 25")
            
                cat = '{}{}{}{}{}{}{}{}'.format(n7,n6,n5,n4,n3,n2,n1,n0)
#                print(cat)
                value = int(cat,2)
                #value = bin_2_signed_int(int(cat,2),32)
                #print(value)
                outarray[index] = value 
                index = index + 1
    print(outarray)
    output_file = open(file_name, mode="wb")
    outarray.tofile(output_file)
    output_file.close()
                

image = read_image(sys.argv[1])
reshape_image(image,sys.argv[2])



