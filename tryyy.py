
class place:
    def __init__(self,x,y,h,w):

        self.xC=x
        self.yC=y
        self.hC=h
        self.wC=w

class obj:
    def __init__(self,place1,object=None,kind=None):

        my_place=place(place1[0],place1[1],place1[2],place1[3])
        self.placeC=my_place
        self.objectC=object
        self.kindC=kind




p=[1,2,3,4]

o=obj(p)

print(o.placeC.hC)
print(o.placeC.xC)