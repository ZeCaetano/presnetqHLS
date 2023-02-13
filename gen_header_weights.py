import aux

f = open("weights.h", "w")
#datatype = "quant_t"
datatype = "quant_reshp"
#values2 = [-1,-2,-3,-4,3,4,5,0,1,2,6,7,-5,-6,-7,-8]
#values3 = [4,5,-6,0,1,2,3,-7,-8,6,7,-1,-2,-3,-4,-5]
#values4 = [-7,-8,6,7,-1,4,5,-3,-4,-5,-6,0,1,2,3,-2]
values2 = ['1111','1110','1101','1100','0011','0100','0101','0000','0001','0010','0110','0111','1011','1010','1001','1000']
values3 = ['0100','0101','1010','0000','0001','0010','0011','1001','1000','0110','0111','1111','1110','1101','1100','1011']
values4 = ['1001','1000','0110','0111','1111','0100','0101','1101','1100','1011','1010','0000','0001','0010','0011','1110']
z1 = 32
nf1 = 48
k1 = 1

z2 = 48
nf2 = 64
#k2 = 1
k2 = 2

z3 = 64
nf3 = 172
k3 = 1

reshp_factor = 4

weights = []


#Init weights
f.write('#include "simple_conv.h"\n\n')
for k in range(z1):
    for i in range(nf1):
        for j in range(k1):
            for l in range(k1):
#                weights[(k*nf1*k1*k1)+(i * k1*k1 + (j * k1) + l)] = values2[(l+j+i+k)%16]
                weights.append(values2[(l+j+i+k)%16])

for k in range(z2):
    for i in range(nf2):
        for j in range(k2):
            for l in range(k2):
#                weights[z1*nf1 + (k*nf2*k2*k2)+(i * k2*k2 + (j * k2) + l)] = values3[(l+j+i+k)%16]
                weights.append(values3[(l+j+i+k)%16])

for k in range(z3):
    for i in range(nf3):
        for j in range(k3):
            for l in range(k3):
#                weights[z1*nf1 + (k*nf2*k2*k2)+(i * k2*k2 + (j * k2) + l)] = values3[(l+j+i+k)%16]
                weights.append(values4[(l+j+i+k)%16])
#                print(values4[(l+j+i+k)%16])

#Write weights in header file
#Weights generation for reshaped weights
f.write(datatype + " weights_l1[{}] = {{".format(z1*nf1*k1*k1//reshp_factor))
for j in range(nf1):
    for i in range(0,z1,4):
        for x in range(k1*k1):
            n0 = weights[(i*k1*k1) + (j*z1*k1*k1) + x]
            n1 = weights[((i+1)*k1*k1) + (j*z1*k1*k1) + x]
            n2 = weights[((i+2)*k1*k1) + (j*z1*k1*k1) + x]
            n3 = weights[((i+3)*k1*k1) + (j*z1*k1*k1) + x]
            cat = '{}{}{}{}'.format(n3,n2,n1,n0)
            val = aux.bin_2_signed_int(int(cat,2),16)            
            f.write(str(val))
#            f.write(str(weights[(i*k1*k1) + (j*z1*k1*k1) + x]))
#            print(str((i*k1*k1) + (j*z1*k1*k1) + x))
            if not((i == z1-4 and j == nf1-1 and x == k1*k1-1)):
                f.write(', ')

f.write('};\n')

#Weights generation for layer 2 with kernel size 1
#f.write(datatype + " weights_l2[{}] = {{".format(z2*nf2*k2*k2))
#for j in range(nf2):
#    for i in range(z2):
#        for x in range(k2*k2):
#            f.write(str(weights[(nf1*z1*k1*k1) + (i*k2*k2) + (j*z2*k2*k2) + x]))
#            print(str((i*k2*k2) + (j*z2*k2*k2) + x))
#            if not((i == z2-1 and j == nf2-1 and x == k2*k2-1)):
#                f.write(', ')

#f.write('};\n')

#Weights generation for layer 2 with kernel size 2
#f.write(datatype + " weights_l2[{}] = {{".format(z2*nf2*k2*k2))
#for j in range(nf2):
#    for x in range(k2):
#        for i in range(z2):
#            for y in range(k2):
#                f.write(str(weights[z1*nf1 + (i*k2*k2) + (j*z2*k2*k2) + x+(y*2)]))
#                print(str((i*k2*k2) + (j*z2*k2*k2) + x+(y*2)))
#                if not((i == z2-1 and j == nf2-1 and x == k2-1 and y == k2-1)):
#                    f.write(', ')
#f.write('};\n')

#Weights generation for layer 2 with kernel size 2 for RESHAPED values
f.write(datatype + " weights_l2[{}] = {{".format(z2*nf2*k2*k2//reshp_factor))
for j in range(nf2):
    for x in range(k2):
        for i in range(0,z2,2):
#            for y in range(k2//(reshp_factor//2)):
            n0 = weights[z1*nf1 + (i*k2*k2) + (j*z2*k2*k2) + x]
#            print(str((i*k2*k2) + (j*z2*k2*k2) + x))
            n1 = weights[z1*nf1 + (i*k2*k2) + (j*z2*k2*k2) + x+2]
#            print(str((i*k2*k2) + (j*z2*k2*k2) + x+2))
            n2 = weights[z1*nf1 + ((i+1)*k2*k2) + (j*z2*k2*k2) + x]
#            print(str(((i+1)*k2*k2) + (j*z2*k2*k2) + x))
            n3 = weights[z1*nf1 + ((i+1)*k2*k2) + (j*z2*k2*k2) + x+2]
#            print(str(((i+1)*k2*k2) + (j*z2*k2*k2) + x+2))
            cat = '{}{}{}{}'.format(n3,n2,n1,n0)
            val = aux.bin_2_signed_int(int(cat,2),16)            
            f.write(str(val))
#                f.write(str(weights[z1*nf1 + (i*k2*k2) + (j*z2*k2*k2) + x+(y*2)]))
#                print(str((i*k2*k2) + (j*z2*k2*k2) + x+(y*2)))
            if not((i == z2-2 and j == nf2-1 and x == k2-1)):
                f.write(', ')
f.write('};\n')

#Weights generation for 3rd layer with kernel size 1
#f.write(datatype + " weights_l3[{}] = {{".format(z3*nf3*k3*k3))
#for j in range(nf3):
#    for i in range(z3):
#        for x in range(k3*k3):
#            f.write(str(weights[(z1*nf1+z2*nf2*k2*k2) + (i*k3*k3) + (j*z3*k3*k3) + x]))
#            print(str((i*k1*k1) + (j*z1*k1*k1) + x))
#            if not((i == z3-1 and j == nf3-1 and x == k3*k3-1)):
#                f.write(', ')

#f.write('};\n')

#Weights generation for 3rd layer with kernel size 1 for RESHAPED values
f.write(datatype + " weights_l3[{}] = {{".format(z3*nf3*k3*k3//reshp_factor))
for j in range(nf3):
    for i in range(0,z3,4):
        for x in range(k3*k3):
            n0 = weights[(z1*nf1+z2*nf2*k2*k2) + (i*k3*k3) + (j*z3*k3*k3) + x]
            n1 = weights[(z1*nf1+z2*nf2*k2*k2) + ((i+1)*k3*k3) + (j*z3*k3*k3) + x]
            n2 = weights[(z1*nf1+z2*nf2*k2*k2) + ((i+2)*k3*k3) + (j*z3*k3*k3) + x]
            n3 = weights[(z1*nf1+z2*nf2*k2*k2) + ((i+3)*k3*k3) + (j*z3*k3*k3) + x]
            cat = '{}{}{}{}'.format(n3,n2,n1,n0)
            val = aux.bin_2_signed_int(int(cat,2),16)            
            f.write(str(val))
#            f.write(str(weights[(z1*nf1+z2*nf2*k2*k2) + (i*k3*k3) + (j*z3*k3*k3) + x]))
#            print(str((i*k1*k1) + (j*z1*k1*k1) + x))
            if not((i == z3-4 and j == nf3-1 and x == k3*k3-1)):
                f.write(', ')

f.write('};\n')

#f.write(datatype + " weights_l2[{}] = {{".format(z2*nf2*k2*k2))
#for j in range(nf2):
#    for i in range(z2):
#        for x in range(k2*k2):
#            print(str((i*k2*k2) + (j*z2*k2*k2) + x))
#            f.write(str(weights[z1*nf1 + (i*k2*k2) + (j*z2*k2*k2) + x]))
#            if not((i == z2-1 and j == nf2-1 and x == k2*k2-1)):
#                f.write(', ')
#f.write('};')


#f.write(datatype + " weights_l1[{}] = {{".format(z1*nf1*k1*k1))
#for i in range(z1):
#    for j in range(nf1):
#        for x in range(k1*k1):
#            f.write(str(weights[(i*k1*k1) + (j*z1*k1*k1) + x]))
#            if not((i == z1-1 and j == nf1-1 and x == k1*k1-1)):
#                f.write(', ')
#
#f.write('};\n')
#
#f.write(datatype + " weights_l2[{}] = {{".format(z2*nf2*k2*k2))
#for i in range(z2):
#    for j in range(nf2):
#        for x in range(k2*k2):
#            f.write(str(weights[z1*nf1 + (i*k2*k2) + (j*z2*k2*k2) + x]))
#            if not((i == z2-1 and j == nf2-1 and x == k2*k2-1)):
#                f.write(', ')
#f.write('};')
f.close
