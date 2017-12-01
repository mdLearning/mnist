'''
Created on Nov 30, 2017

@author: lyf
'''
#-*- coding: utf-8 -*-
from PIL import Image
import struct

def read_image(filename):
    f = open(filename,'rb')
    index = 0
    buf = f.read()
    f.close()
    magic,images,rows,columns = struct.unpack_from('>IIII',buf,index)
    index += struct.calcsize('>IIII')
    for i in range(images):
        image = Image.new('L',(columns,rows))
        for x in xrange(rows):
            for y in xrange(columns):
                image.putpixel((y,x),int(struct.unpack_from('>B',buf,index)[0]))
                index += struct.calcsize('>B')
        print 'save' + str(i) + 'image'
        image.save(str(i) + '.png')
        
def read_label(filename,saveFilename):
    f = open(filename,'rb')
    index = 0
    buf = f.read()
    f.close()
    magic,labels = struct.unpack_from('II',buf,index)
    index += struct.calcsize('II')
    labelArr = [0]*labels
    for x in xrange(labels):
        lableArr[x] = int(struct.unpack_from('>B',buf,index)[0])
        index += struct.calcsize('>B')
    save = open(saveFilename,'w')
    save.write(','.join(map(lambda x:str(x),labelArr)))
    save.write('\n')
    save.close()
    print 'save labels success'

if __name__=='__main__':
    read_image('/Users/lyf/Documents/ML_exercise/mnist/dataset/t10k-images-idx3-ubyte')
    read_label('/Users/lyf/Documents/ML_exercise/mnist/dataset/t10k-labels-idx1-ubyte','test/label.txt')