import numpy as np
import os
import platform
import cv2 as cv
import pandas as pd
import copy
import random
from scipy.stats import bernoulli, binom

####################################################################
# Dijkstra's algorithm (used for constructing syndrome node graph) #
####################################################################
##                                                                ##

def modulo(arr,n):
    for x in range(len(arr)):
        for y in range(len(arr[0])):
            arr[x,y] = int(arr[x,y]) % int(n)
    return arr
            
def minDistance(dist,queue):
    minimum = float("Inf")
    min_index = -1
    for i in range(len(dist)):
        if dist[i] < minimum and i in queue:
            minimum = dist[i]
            min_index = i
    return min_index

def printPath(parent, j):
    if parent[j] == -1 :
        print(j, end =" ")
        return
    printPath(parent , parent[j])
    print (j, end =" ")  

def dijkstra(graph, src):
 
    row = len(graph)
    col = len(graph[0])

    dist = [float("Inf")] * row
    parent = [-1] * row
    dist[src] = 0

    # Add all vertices in queue
    queue = []
    for i in range(row):
        queue.append(i)

    while queue:
        u = minDistance(dist,queue) 
        queue.remove(u)
        for i in range(col):
            '''Update dist[i] only if it is in queue, there is
            an edge from u to i, and total weight of path from
            src to i through u is smaller than current value of
            dist[i]'''
            if graph[u][i] and i in queue:
                if dist[u] + graph[u][i] < dist[i]:
                    dist[i] = dist[u] + graph[u][i]
                    parent[i] = u

    # print the constructed distance array
    print("Vertex \t\tDistance from Source\t\tPath")
    for i in range(1, len(dist)):
        print("\n%d --> %d \t\t%f \t\t" % (src, i, dist[i]), end =" "),
        printPath(parent,i)

##                                                                ##
####################################################################
####################################################################

class D_Lattice:
    def __init__(self,width,length,holes):
        self.width=width
        self.length=length
        self.lattice = np.zeros((2*width+1,2*length+1))
        self.x_errors_on_lattice = np.zeros((2*width+1,2*length+1))
        self.z_errors_on_lattice = np.zeros((2*width+1,2*length+1))
        self.X_stabilizer_graph = np.ones(( (self.length+1)*(self.width+1) ,
                                            (self.length+1)*(self.width+1) ))*10000
        self.Z_stabilizer_graph = np.ones(( (self.length)*(self.width)+2*(self.width+self.length),
                                      (self.length)*(self.width)+2*(self.width+self.length)  ))*10000
        self.original_X_stabilizer_graph = np.ones(( (self.length+1)*(self.width+1) ,
                                            (self.length+1)*(self.width+1) ))*10000
        self.original_Z_stabilizer_graph = np.ones(( (self.length)*(self.width)+2*(self.width+self.length),
                                      (self.length)*(self.width)+2*(self.width+self.length)  ))*10000
        self.marked_x_stabilzer=[]
        self.marked_z_stabilzer=[]
        self.x_parents=[]
        self.z_parents=[]
        self.closest_boundary_nodes=[] # For dual lattice 
        self.x_corretion_on_lattice = np.zeros((2*self.width+1,2*self.length+1))
        self.z_corretion_on_lattice = np.zeros((2*self.width+1,2*self.length+1))
        self.x_corretion_result_on_lattice = np.zeros((2*self.width+1,2*self.length+1))
        self.z_corretion_result_on_lattice = np.zeros((2*self.width+1,2*self.length+1))
        self.holes=holes
        for h in holes:
            assert h[0]>0 and h[1]>0, "Invalid Hole position."
            assert h[0]<2*length and h[1]<2*width, "Invalid Hole position."

            self.lattice[h[1],h[0]]=-1
    
    def clear_lattice(self):
        self.lattice = np.zeros((2*self.width+1,2*self.length+1))
        for h in self.holes:
            assert h[0]>0 and h[1]>0, "Invalid Hole position."
            assert h[0]<2*self.length and h[1]<2*self.width, "Invalid Hole position."
            self.lattice[h[1],h[0]]=-1
        self.x_errors_on_lattice = np.zeros((2*self.width+1,2*self.length+1))
        self.z_errors_on_lattice = np.zeros((2*self.width+1,2*self.length+1))
        self.X_stabilizer_graph = copy.copy(self.original_X_stabilizer_graph)
        self.Z_stabilizer_graph = copy.copy(self.original_Z_stabilizer_graph)
        self.marked_x_stabilzer=[]
        self.marked_z_stabilzer=[]
        self.x_parents=[]
        self.z_parents=[]
        self.closest_boundary_nodes=[]
        self.x_corretion_on_lattice = np.zeros((2*self.width+1,2*self.length+1))
        self.z_corretion_on_lattice = np.zeros((2*self.width+1,2*self.length+1))
        self.x_corretion_result_on_lattice = np.zeros((2*self.width+1,2*self.length+1))
        self.z_corretion_result_on_lattice = np.zeros((2*self.width+1,2*self.length+1))
    
    def PositionIsQubit(self,q):
        assert q[0]<=2*self.length and q[0]>=0,"PositionIsQubit(): q[0] is out of range"
        assert q[1]<=2*self.width and q[1]>=0,"PositionIsQubit(): q[1] is out of range"
        if( (q[0]+q[1])%2==1 ): return True;
        else: return False;
        
    def PositionIsZStabilizer(self,q):
        assert q[0]<=2*self.length and q[0]>=0,"PositionIsZStabilizer(): q[0] is out of range"
        assert q[1]<=2*self.width and q[1]>=0,"PositionIsZStabilizer(): q[1] is out of range"
        if( (q[0]+q[1])%2==0 and q[0]%2==1 ): return True;
        else: return False;
    
    def PositionIsXStabilizer(self,q):
        assert q[0]<=2*self.length and q[0]>=0,"PositionIsXStabilizer: q[0] is out of range"
        assert q[1]<=2*self.width and q[1]>=0,"PositionIsXStabilizer: q[1] is out of range"
        if( (q[0]+q[1])%2==0 and q[0]%2==0 ): return True;
        else: return False;
    
    def PositionIsHole(self,q):
        assert q[0]<=2*self.length and q[0]>=0,"PositionIsQubit(): q[0] is out of range"
        assert q[1]<=2*self.width and q[1]>=0,"PositionIsQubit(): q[1] is out of range"
        if self.lattice[q[1],q[0]]==-1 : return True;
        else: return False;

    def Receiving_syndrome(self,syndrome_nodes):
        for syndorme_node in syndrome_nodes:
            assert self.PositionIsXStabilizer(syndorme_node) or self.PositionIsZStabilizer(syndorme_node), "Receiving_syndrome(): Invalid syndrome position."
            assert not self.PositionIsHole(syndorme_node), "Receiving_syndrome(): The input syndrome is in a hole."
            self.lattice[syndorme_node[1],syndorme_node[0]]=( self.lattice[syndorme_node[1],syndorme_node[0]] + 1 )%2
    
    # correnction: 0 = correnction, 1 = errors, 2 = errors + correction
    def show_lattice(self,syndrome=1,correnction=1):
        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)
        RED = (0, 0, 255)
        GREEN = (0, 255, 0)
        BLUE = (255, 0, 0)
        assert self.width>0 and self.length>0, "Width and length need to be greater than 0."
        if self.width<=9 or self.length<=9 : interval = 50
        else : interval = int(400/max(self.width, self.length))
        img = np.ones(( (self.width*2)*interval+interval*2, (self.length*2)*interval+interval*2, 3), np.uint8)*255
        
        # Plot the grid
        for y in range(0,self.width+1):
            cv.line(img,(interval,y*2*interval+interval),(self.length*2*interval+interval,y*2*interval+interval),BLACK,3)
        for x in range(0,self.length+1):       
            cv.line(img,(x*2*interval+interval,interval),(x*2*interval+interval,self.width*2*interval+interval),BLACK,3)
        
        # Plot the dual lattice
        for y in range(1,self.width+1):
            x=interval
            while x+10<=self.length*2*interval+interval:
                cv.line(img,(x,y*2*interval),(x+10,y*2*interval),BLACK,2)
                x=x+20
        
        for x in range(1,self.length+1):
            y=interval
            while y+10<=self.width*2*interval+interval:
                cv.line(img,(x*2*interval,y),(x*2*interval,y+10),BLACK,2)
                y=y+20
        
        # Plot holes
        for h in self.holes:
            assert h[0]<=2*self.length and h[0]>=0,"x-position is out of range"
            assert h[1]<=2*self.width and h[1]>=0,"y-position is out of range"
            if (h[0]+h[1])%2 == 0:
                cv.rectangle(img, ( (h[0]-1)*interval+interval, (h[1]-1)*interval+interval), 
                                  ( (h[0]+1)*interval+interval, (h[1]+1)*interval+interval),BLACK,-1)

        # Plot the syndrome
        assert syndrome==0 or syndrome==1, "No such syndrome option."
        if syndrome==1:
            for x in range(0,self.length*2+1):
                for y in range(0,self.width*2+1):
                    if (x+y)%2==0 and self.lattice[y,x]==1:
                        # Z-type error
                        if x%2==0 : cv.circle(img, (x*interval+interval,y*interval+interval), 10, BLUE, -1)
                        # X-type error
                        if x%2==1 : cv.circle(img, (x*interval+interval,y*interval+interval), 10, RED, -1)
        
        # Plot the correnction/errors
        # 0: Nothing
        # 1: correction
        # 2: errors
        # 3: correction + errors
        assert correnction==0 or correnction==1 or correnction==2 or correnction==3, "No such correnction option."
        if correnction==1:
            for x in range(0,self.length*2+1):
                for y in range(0,self.width*2+1):
                    # X-type correction
                    if (x+y)%2==1 and self.x_corretion_on_lattice[y,x]==1:
                        if x%2==1: cv.line(img,(x*interval+interval,(y-1)*interval+interval) , 
                                               (x*interval+interval,(y+1)*interval+interval), RED, 5)
                        if x%2==0: cv.line(img,((x-1)*interval+interval,y*interval+interval) , 
                                               ((x+1)*interval+interval,y*interval+interval), RED, 5)
                    # Z-type error
                    if (x+y)%2==1 and self.z_corretion_on_lattice[y,x]==1:
                        if x%2==1: cv.line(img,((x-1)*interval+interval,y*interval+interval), 
                                               ((x+1)*interval+interval,y*interval+interval), BLUE, 5)
                        if x%2==0: cv.line(img,(x*interval+interval,(y-1)*interval+interval), 
                                               (x*interval+interval,(y+1)*interval+interval), BLUE, 5)
        if correnction==2:
            for x in range(0,self.length*2+1):
                for y in range(0,self.width*2+1):
                    # X-type error
                    if (x+y)%2==1 and self.x_errors_on_lattice[y,x]==1:
                        if x%2==1: cv.line(img,(x*interval+interval,(y-1)*interval+interval) , 
                                               (x*interval+interval,(y+1)*interval+interval), RED, 5)
                        if x%2==0: cv.line(img,((x-1)*interval+interval,y*interval+interval) , 
                                               ((x+1)*interval+interval,y*interval+interval), RED, 5)
                    # Z-type error
                    if (x+y)%2==1 and self.z_errors_on_lattice[y,x]==1:
                        if x%2==1: cv.line(img,((x-1)*interval+interval,y*interval+interval), 
                                               ((x+1)*interval+interval,y*interval+interval), BLUE, 5)
                        if x%2==0: cv.line(img,(x*interval+interval,(y-1)*interval+interval), 
                                               (x*interval+interval,(y+1)*interval+interval), BLUE, 5)
        
        if correnction==3:
            for x in range(0,self.length*2+1):
                for y in range(0,self.width*2+1):
                    # X-type correction
                    if (x+y)%2==1 and self.x_corretion_result_on_lattice[y,x]==1:
                        if x%2==1: cv.line(img,(x*interval+interval,(y-1)*interval+interval) , 
                                               (x*interval+interval,(y+1)*interval+interval), RED, 5)
                        if x%2==0: cv.line(img,((x-1)*interval+interval,y*interval+interval) , 
                                               ((x+1)*interval+interval,y*interval+interval), RED, 5)
                    # Z-type error
                    if (x+y)%2==1 and self.z_corretion_result_on_lattice[y,x]==1:
                        if x%2==1: cv.line(img,((x-1)*interval+interval,y*interval+interval), 
                                               ((x+1)*interval+interval,y*interval+interval), BLUE, 5)
                        if x%2==0: cv.line(img,(x*interval+interval,(y-1)*interval+interval), 
                                               (x*interval+interval,(y+1)*interval+interval), BLUE, 5)
                
        cv.imshow('Lattice', img) 
        cv.waitKey(0)
        
    def single_z(self,position):
        assert position[0]<=2*self.length and position[0]>=0,"x-position is out of range"
        assert position[1]<=2*self.width and position[1]>=0,"y-position is out of range"
        assert (position[0]+position[1])%2 != 0 , "The input position is not a qubit"
        if self.lattice[position[1],position[0]] != -1:
            self.z_errors_on_lattice[position[1],position[0]] = \
                (self.z_errors_on_lattice[position[1],position[0]]+1)%2
            if position[0]%2 == 0:
                self.lattice[position[1]-1,position[0]]=( self.lattice[position[1]-1,position[0]] + 1 )%2
                self.lattice[position[1]+1,position[0]]=( self.lattice[position[1]+1,position[0]] + 1 )%2
            elif position[1]%2 == 0:
                self.lattice[position[1],position[0]-1]=( self.lattice[position[1],position[0]-1] + 1 )%2
                self.lattice[position[1],position[0]+1]=( self.lattice[position[1],position[0]+1] + 1 )%2
    
    def single_x(self,position):
        assert position[0]<=2*self.length and position[0]>=0,"x-position is out of range"
        assert position[1]<=2*self.width and position[1]>=0,"y-position is out of range"
        assert (position[0]+position[1])%2 != 0 , "The input position is not a qubit"
        if self.lattice[position[1],position[0]] != -1:
            self.x_errors_on_lattice[position[1],position[0]]=(self.x_errors_on_lattice[position[1],position[0]]+1)%2
            if position[0]%2 == 0:
                if position[0]-1 >= 0 and self.lattice[position[1],position[0]-1] != -1 : 
                    self.lattice[position[1],position[0]-1]=( self.lattice[position[1],position[0]-1] + 1 )%2
                if position[0]+1 <= 2*self.length and self.lattice[position[1],position[0]+1] !=-1 :
                    self.lattice[position[1],position[0]+1]=( self.lattice[position[1],position[0]+1] + 1 )%2
            elif position[1]%2 == 0:
                if position[1]-1 >= 0 and self.lattice[position[1]-1,position[0]]!=-1: 
                    self.lattice[position[1]-1,position[0]]=( self.lattice[position[1]-1,position[0]] + 1 )%2
                if position[1]+1 <= 2*self.width and self.lattice[position[1]+1,position[0]]!=-1 : 
                    self.lattice[position[1]+1,position[0]]=( self.lattice[position[1]+1,position[0]] + 1 )%2
    
    # noise_model: 0=uncorrelated noise model, 1=depolarizing noise model  
    def apply_noises(self,prob=0.1,noise_model=0):
        assert noise_model==0 or noise_model==1, "You've chosen a wrong error model."
        qubits=[]
        NumberOfQubits=0
        for x in range(0,self.length*2+1):
            for y in range(0,self.width*2+1):
                if (x+y)%2==1 and self.lattice[y,x]!=-1:
                    qubits.append((x,y))
                    NumberOfQubits += 1

        if noise_model == 0:
            X_error_set = bernoulli(prob).rvs(NumberOfQubits)
            Z_error_set = bernoulli(prob).rvs(NumberOfQubits)
            for e in zip(qubits,X_error_set):
                if e[1]==1: self.single_x(e[0])
            for e in zip(qubits,Z_error_set):
                if e[1]==1: self.single_z(e[0])
        
        elif noise_model==1:
            error_set = bernoulli(prob).rvs(NumberOfQubits)
            for e in zip(qubits,error_set):
                if e[1]==1:
                    temp = random.randint(1,3)
                    if temp==1: self.single_x(e[0])
                    elif temp==2: self.single_z(e[0])
                    elif temp==3:
                        self.single_x(e[0])
                        self.single_z(e[0])

    def string_Z_correction(self,parent,src,dst):
        current_mark=dst
        while current_mark!=src:
            next_mark=parent[current_mark]
            y_c = 2*( int( current_mark/(self.length+1) ) )
            x_c = 2*( int( current_mark%(self.length+1) ) )
            y_n = 2*( int( next_mark/(self.length+1) ) )
            x_n = 2*( int( next_mark%(self.length+1) ) )
            assert y_c==y_n or x_c==x_n, "string_Z_correction(): The edge between the two marks doesn't exist."
            assert y_c!=y_n or x_c!=x_n, "string_Z_correction(): The two marks are the same."
            if y_c==y_n:
                assert x_c==x_n+2 or x_c==x_n-2, "string_Z_correction(): The two marks are not next two each other."
                self.z_corretion_on_lattice[y_c,int( (x_c+x_n)/2 )]=1
            elif x_c==x_n:
                assert y_c==y_n+2 or y_c==y_n-2, "string_correction(): The two marks are not next two each other."
                self.z_corretion_on_lattice[int( (y_c+y_n)/2 ), x_c]=1
            current_mark=next_mark
            
    def string_X_correction(self,parent,src,dst):
        Number_Z_Stabilizer = (self.length)*(self.width)
        current_mark=dst
        if current_mark < Number_Z_Stabilizer:
            y_c = 2*( int( current_mark/(self.length) ) ) + 1
            x_c = 2*( current_mark%self.length ) + 1
        elif current_mark < Number_Z_Stabilizer+self.length:
            y_c = -1
            x_c = 2*( current_mark - Number_Z_Stabilizer ) + 1
        elif current_mark < Number_Z_Stabilizer + self.length + self.width:
            y_c = 2*( current_mark - (Number_Z_Stabilizer+self.length)  ) + 1
            x_c = -1
        elif current_mark < Number_Z_Stabilizer + self.length + self.width*2:
            y_c = 2*( current_mark - (Number_Z_Stabilizer+self.length + self.width)  ) + 1
            x_c = self.length*2+1
        else:
            y_c = 2*self.width+1
            x_c = 2*( current_mark - (Number_Z_Stabilizer+self.length + self.width*2) ) + 1
        while current_mark!=src:
            next_mark=parent[current_mark]
            y_n = 2*( int( next_mark/self.length ) ) + 1
            x_n = 2*(  next_mark%self.length  ) + 1
            #if not (y_c==y_n or x_c==x_n) :
            #  print(dst,current_mark,next_mark,src)
            #  print(x_c,y_c,x_n,y_n)
            #if not (y_c!=y_n or x_c!=x_n):
            #  print(dst,current_mark,next_mark,src)
            #  print(x_c,y_c,x_n,y_n)
            assert y_c==y_n or x_c==x_n, "string_X_correction(): The edge between the two marks doesn't exist."
            assert y_c!=y_n or x_c!=x_n, "string_X_correction(): The two marks are the same."
            if y_c==y_n:
                assert x_c==x_n+2 or x_c==x_n-2, "string_X_correction(): The two marks are not next two each other."
                self.x_corretion_on_lattice[y_c,int( (x_c+x_n)/2 )]=1
            elif x_c==x_n:
                assert y_c==y_n+2 or y_c==y_n-2, "string_X_correction(): The two marks are not next two each other."
                self.x_corretion_on_lattice[int( (y_c+y_n)/2 ), x_c]=1
            current_mark=next_mark
            y_c=y_n
            x_c=x_n
    
    def Construct_X_stabilizer_graph(self):
        Number_X_Stabilizer = (self.length+1)*(self.width+1)  ## including holes
        self.X_stabilizer_graph = np.ones((Number_X_Stabilizer,Number_X_Stabilizer))*10000
        np.fill_diagonal(self.X_stabilizer_graph,0)
        for x in range(0,self.length+1):
            for y in range(0,self.width+1):
                if x!=self.length and self.lattice[ y*2 , x*2 + 1 ] != -1:
                    self.X_stabilizer_graph[ y*(self.length+1)+x+1,  y*(self.length+1)+x  ] = 2
                    self.X_stabilizer_graph[ y*(self.length+1)+x, y*(self.length+1)+x+1 ] = 2
                if(y!=self.width) and self.lattice[ y*2+1 , x*2 ] != -1:
                    self.X_stabilizer_graph[ (y+1)*(self.length+1)+x,  y*(self.length+1)+x  ] = 2
                    self.X_stabilizer_graph[ y*(self.length+1)+x, (y+1)*(self.length+1)+x ] = 2
        
        for h in self.holes:
            if self.PositionIsXStabilizer(h):
                x=int( h[0]/2 )
                y=int( h[1]/2 )
                if(y!=0):
                    self.X_stabilizer_graph[ (y-1)*(self.length+1)+x ,  y*(self.length+1)+x  ] = 0
                    self.X_stabilizer_graph[  y*(self.length+1)+x , (y-1)*(self.length+1)+x  ] = 0
                if(x!=0):
                    self.X_stabilizer_graph[ y*(self.length+1)+x-1,  y*(self.length+1)+x  ] = 0
                    self.X_stabilizer_graph[ y*(self.length+1)+x, y*(self.length+1)+x-1 ] = 0
                if(x!=self.length):
                    self.X_stabilizer_graph[ y*(self.length+1)+x+1,  y*(self.length+1)+x  ] = 0
                    self.X_stabilizer_graph[ y*(self.length+1)+x, y*(self.length+1)+x+1 ] = 0
                if(y!=self.width):
                    self.X_stabilizer_graph[ (y+1)*(self.length+1)+x,  y*(self.length+1)+x  ] = 0
                    self.X_stabilizer_graph[ y*(self.length+1)+x, (y+1)*(self.length+1)+x ] = 0
        np.set_printoptions(suppress=True)
        self.original_X_stabilizer_graph = copy.copy(self.X_stabilizer_graph)
    
    def Construct_marked_X_graph(self):
        #######################################################################
        ############# Dijkstra's Algorithm on X_stabilizer_graph ###############
        #######################################################################
        
        self.marked_x_stabilzer=[]
        for m in range(len(self.X_stabilizer_graph)):
            if self.lattice[2*( int( m/(self.length+1) ) ),2*( int( m%(self.length+1) ) )]==1:
                self.marked_x_stabilzer.append(m)
        f=open("syndrome_node_graph.txt","w")
        f.write( "%d\n" % len(self.marked_x_stabilzer) ) 
        f.write( "%d\n" % int(len(self.marked_x_stabilzer)*(len(self.marked_x_stabilzer)-1)/2) ) 
        self.x_parents=[]
        for d in range(len(self.marked_x_stabilzer)):
            src = self.marked_x_stabilzer[d]
            row = len(self.X_stabilizer_graph)
            col = len(self.X_stabilizer_graph[0])
            dist = [float("Inf")] * row
            parent = [-1] * row
            dist[src] = 0
            # Add all vertices in queue
            queue = []
            for j in range(row):
                queue.append(j)

            while queue:
                u = minDistance(dist,queue) 
                queue.remove(u)
                for i in range(col):
                    if self.X_stabilizer_graph[u][i] and i in queue:
                        if dist[u] + self.X_stabilizer_graph[u][i] < dist[i]:
                            dist[i] = dist[u] + self.X_stabilizer_graph[u][i]
                            parent[i] = u

            self.x_parents.append(parent)
            # print the constructed distance array
            for s in range(d+1,len(self.marked_x_stabilzer)):
                f.write("%d %d %f\n" % (d, s, dist[self.marked_x_stabilzer[s]]))
                #printPath(parent,marked_x_stabilzer[s])
                      
    def Reweight_X_stabilizer_graph(self):
        self.X_stabilizer_graph = copy.copy(self.original_X_stabilizer_graph)
        for x in range(0,self.length+1):
            for y in range(0,self.width+1):
                if x!=self.length and self.x_corretion_on_lattice[ y*2 , x*2 + 1 ] == 1:
                    self.X_stabilizer_graph[ y*(self.length+1)+x+1,  y*(self.length+1)+x  ] = 0.01 
                    self.X_stabilizer_graph[ y*(self.length+1)+x, y*(self.length+1)+x+1 ] = 0.01
                if(y!=self.width) and self.x_corretion_on_lattice[ y*2+1 , x*2 ] == 1:
                    self.X_stabilizer_graph[ (y+1)*(self.length+1)+x,  y*(self.length+1)+x  ] = 0.01
                    self.X_stabilizer_graph[ y*(self.length+1)+x, (y+1)*(self.length+1)+x ] = 0.01
        
    def Construct_Z_stabilizer_graph(self,prob=0.1):
        Number_Z_Stabilizer = (self.length)*(self.width)  ## including holes
        self.Z_stabilizer_graph = np.ones(( Number_Z_Stabilizer+2*(self.width+self.length),
                                      Number_Z_Stabilizer+2*(self.width+self.length)  ))*10000
        np.fill_diagonal(self.Z_stabilizer_graph,0)
        
        for x in range(0,self.length):
            for y in range(0,self.width):
                if(x!=self.length-1):
                    self.Z_stabilizer_graph[ y*(self.length)+x+1,  y*(self.length)+x  ] = 2
                    self.Z_stabilizer_graph[ y*(self.length)+x, y*(self.length)+x+1 ] = 2
                if(y!=self.width-1):
                    self.Z_stabilizer_graph[ (y+1)*(self.length)+x,  y*(self.length)+x  ] = 2
                    self.Z_stabilizer_graph[ y*(self.length)+x, (y+1)*(self.length)+x ] = 2
        
        for x in range(0,self.length):
            self.Z_stabilizer_graph[ x,  x+Number_Z_Stabilizer  ] = 2
            self.Z_stabilizer_graph[ x+Number_Z_Stabilizer,  x  ] = 2
            self.Z_stabilizer_graph[ x+self.length*(self.width-1),
                               x+Number_Z_Stabilizer+self.length+2*self.width ] = 2
            self.Z_stabilizer_graph[ x+Number_Z_Stabilizer+self.length+2*self.width, 
                               x+self.length*(self.width-1)  ] = 2
        
        for y in range(0,self.width):
            self.Z_stabilizer_graph[ y*self.length,  y+Number_Z_Stabilizer+self.length  ] = 2
            self.Z_stabilizer_graph[ y+Number_Z_Stabilizer+self.length, y*self.length   ] = 2
            self.Z_stabilizer_graph[ y*self.length+self.length-1,  
                               y+Number_Z_Stabilizer+self.length+self.width  ] = 2
            self.Z_stabilizer_graph[ y+Number_Z_Stabilizer+self.length+self.width, 
                               y*self.length+self.length-1   ] = 2
        
        for h in self.holes:
            if self.PositionIsZStabilizer(h):
                x=int(h[0]/2)
                y=int(h[1]/2)
                self.lattice[h[1],h[0]]=-1
                if y!=0 and self.PositionIsHole((h[0],h[1]-2)):
                    self.Z_stabilizer_graph[ (y-1)*(self.length)+x,  y*(self.length)+x  ] = 0
                    self.Z_stabilizer_graph[ y*(self.length)+x, (y-1)*(self.length)+x ] = 0
                if x!=0 and self.PositionIsHole((h[0]-2,h[1])):
                    self.Z_stabilizer_graph[ y*(self.length)+x-1,  y*(self.length)+x  ] = 0
                    self.Z_stabilizer_graph[ y*(self.length)+x, y*(self.length)+x-1 ] = 0
                if x!=self.length-1 and self.PositionIsHole((h[0]+2,h[1])):
                    self.Z_stabilizer_graph[ y*(self.length)+x+1,  y*(self.length)+x  ] = 0
                    self.Z_stabilizer_graph[ y*(self.length)+x, y*(self.length)+x+1 ] = 0
                if y!=self.width-1 and self.PositionIsHole((h[0],h[1]+2)):
                    self.Z_stabilizer_graph[ (y+1)*(self.length)+x,  y*(self.length)+x  ] = 0
                    self.Z_stabilizer_graph[ y*(self.length)+x, (y+1)*(self.length)+x ] = 0
        np.set_printoptions(suppress=True)
        self.original_Z_stabilizer_graph = copy.copy(self.Z_stabilizer_graph)
        #np.savetxt("Z_stabilizer_graph.csv",self.Z_stabilizer_graph, delimiter=',', fmt='%f')
        
    def Construct_marked_Z_graph(self,prob=0.1):
        #######################################################################
        ############# Dijkstra's Algorithm on Z_stabilizer_graph ###############
        #######################################################################
        Number_Z_Stabilizer = (self.length)*(self.width)
        self.marked_z_stabilzer=[]
        for m in range(Number_Z_Stabilizer):
            if self.lattice[ 2*( int( m/(self.length) ) )+1 , 2*( int( m%(self.length) ) )+1 ]==1:
                self.marked_z_stabilzer.append(m);
        f=open("syndrome_node_graph.txt","w")
        f.write( "%d\n" % (len(self.marked_z_stabilzer)*2) )
        num_edges = len(self.marked_z_stabilzer)*(len(self.marked_z_stabilzer)-1)+len(self.marked_z_stabilzer)
        f.write( "%d\n" %  num_edges ) 
        self.z_parents=[]
        self.closest_boundary_nodes=[]
        for d in range(len(self.marked_z_stabilzer)):
            src = self.marked_z_stabilzer[d]
            row = len(self.Z_stabilizer_graph)
            col = len(self.Z_stabilizer_graph[0])
            dist = [float("Inf")] * row
            parent = [-1] * row
            dist[src] = 0
            # Add all vertices in queue
            queue = []
            for j in range(row):
                queue.append(j)

            while queue:
                u = minDistance(dist,queue) 
                queue.remove(u)
                for i in range(col):
                    if self.Z_stabilizer_graph[u][i] and i in queue:
                        if dist[u] + self.Z_stabilizer_graph[u][i] < dist[i]:
                            dist[i] = dist[u] + self.Z_stabilizer_graph[u][i]
                            parent[i] = u

            self.z_parents.append(parent)
            # print the constructed distance array
            for s in range(d+1, len(self.marked_z_stabilzer)):
                f.write("%d %d %f\n" % (d, s, dist[self.marked_z_stabilzer[s]]))
                #printPath(parent,marked_x_stabilzer[s])
            
            closest_boundary_node=-1
            distance_to_boundary = 10000
            for h in self.holes:
                if (h[0]+h[1])%2==0 and h[0]%2==1:
                    if dist[ int(h[0]/2) + int(h[1]/2)*self.length  ] < distance_to_boundary: 
                        distance_to_boundary = dist[ int(h[0]/2) + int(h[1]/2)*self.length  ]
                        closest_boundary_node = int(h[0]/2) + int(h[1]/2)*self.length
            
            for b in range(Number_Z_Stabilizer, Number_Z_Stabilizer+2*(self.width+self.length) ):
                if dist[b] < distance_to_boundary:
                    distance_to_boundary = dist[b]
                    closest_boundary_node = b
            self.closest_boundary_nodes.append(closest_boundary_node)                             
            f.write("%d %d %f\n" % (d, len(self.marked_z_stabilzer)+d, distance_to_boundary))
            
        for b1 in range(len(self.marked_z_stabilzer),len(self.marked_z_stabilzer)*2):
            for b2 in range(b1+1,len(self.marked_z_stabilzer)*2):
                f.write("%d %d %f\n" % (b1, b2, 0))
    
    def Reweight_Z_stabilizer_graph(self):
        Number_Z_Stabilizer = (self.length)*(self.width)
        self.Z_stabilizer_graph = copy.copy(self.original_Z_stabilizer_graph)
        for x in range(0,self.length):
            for y in range(0,self.width):
                if (x!=self.length-1) and self.z_corretion_on_lattice[ 2*y+1 , 2*x+1+1]==1 :
                    self.Z_stabilizer_graph[ y*(self.length)+x+1,  y*(self.length)+x  ] = 0.01
                    self.Z_stabilizer_graph[ y*(self.length)+x, y*(self.length)+x+1 ] = 0.01
                if(y!=self.width-1) and self.z_corretion_on_lattice[ 2*y+1+1 , 2*x+1]==1 :
                    self.Z_stabilizer_graph[ (y+1)*(self.length)+x,  y*(self.length)+x  ] = 0.01
                    self.Z_stabilizer_graph[ y*(self.length)+x, (y+1)*(self.length)+x ] = 0.01
        
        for x in range(0,self.length):
            if self.z_corretion_on_lattice[ 0 , 2*x+1]==1:
                self.Z_stabilizer_graph[ x,  x+Number_Z_Stabilizer  ] = 0.01
                self.Z_stabilizer_graph[ x+Number_Z_Stabilizer,  x  ] = 0.01
            if self.z_corretion_on_lattice[  2*self.width , 2*x+1]==1:
                self.Z_stabilizer_graph[ x+self.length*(self.width-1),
                                   x+Number_Z_Stabilizer+self.length+2*self.width ] = 0.01
                self.Z_stabilizer_graph[ x+Number_Z_Stabilizer+self.length+2*self.width, 
                               x+self.length*(self.width-1)  ] = 0.01
        
        for y in range(0,self.width):
            if self.z_corretion_on_lattice[ 2*y+1 , 0]==1:
                self.Z_stabilizer_graph[ y*self.length,  y+Number_Z_Stabilizer+self.length  ] = 0.01
                self.Z_stabilizer_graph[ y+Number_Z_Stabilizer+self.length, y*self.length   ] = 0.01
            if self.z_corretion_on_lattice[ 2*y+1 , 2*self.length]==1:
                self.Z_stabilizer_graph[ y*self.length+self.length-1,  
                                   y+Number_Z_Stabilizer+self.length+self.width  ] = 0.01
                self.Z_stabilizer_graph[ y+Number_Z_Stabilizer+self.length+self.width, 
                               y*self.length+self.length-1   ] = 0.01
    
    def Z_correction(self):
        correction_pairs = []
        self.z_corretion_on_lattice = np.zeros((2*self.width+1,2*self.length+1))
        self.z_corretion_result_on_lattice = np.zeros((2*self.width+1,2*self.length+1))
        with open("matching_results.txt", "r", encoding = "utf-8") as fp:
            for i in fp.readlines():
                tmp = i.split(" ")
                correction_pairs.append([int(tmp[0]), int(tmp[1])])
        for p in correction_pairs:
            self.string_Z_correction(self.x_parents[p[0]],self.marked_x_stabilzer[p[0]],self.marked_x_stabilzer[p[1]])
        self.z_corretion_result_on_lattice = self.z_errors_on_lattice + self.z_corretion_on_lattice
        self.z_corretion_result_on_lattice = modulo(self.z_corretion_result_on_lattice,2)
    
    def X_correction(self):
        correction_pairs = []
        self.x_corretion_on_lattice = np.zeros((2*self.width+1,2*self.length+1))
        self.x_corretion_result_on_lattice = np.zeros((2*self.width+1,2*self.length+1))
        with open("matching_results.txt", "r", encoding = "utf-8") as fp:
            for i in fp.readlines():
                tmp = i.split(" ")
                correction_pairs.append([int(tmp[0]), int(tmp[1])])
        for p in correction_pairs:
            assert p[0] < len(self.marked_z_stabilzer)*2 and p[1] < len(self.marked_z_stabilzer)*2, \
                "X_correction: The input pair is wrong"
            if p[0] < len(self.marked_z_stabilzer):
                if p[1] < len(self.marked_z_stabilzer):
                    self.string_X_correction(self.z_parents[p[0]],self.marked_z_stabilzer[p[0]],self.marked_z_stabilzer[p[1]])
                else:
                    self.string_X_correction(self.z_parents[p[0]], 
                                             self.marked_z_stabilzer[p[0]] , 
                                             self.closest_boundary_nodes[p[1]-len(self.marked_z_stabilzer)])
        self.x_corretion_result_on_lattice = self.x_errors_on_lattice + self.x_corretion_on_lattice
        self.x_corretion_result_on_lattice = modulo(self.x_corretion_result_on_lattice,2)
        
    def single_Z_stabilizer_on_Z_result_lattice(self,z_pos):
        assert self.PositionIsZStabilizer(z_pos), \
            "single_Z_stabilizer_on_Z_result_lattice(): The input is not a Z-stabilizer"
        if self.PositionIsHole(z_pos)==False:
            self.z_corretion_result_on_lattice[ z_pos[1]+1 , z_pos[0] ] = \
                (self.z_corretion_result_on_lattice[ z_pos[1]+1 , z_pos[0] ] + 1 )%2
            self.z_corretion_result_on_lattice[ z_pos[1]-1 , z_pos[0] ] = \
                (self.z_corretion_result_on_lattice[ z_pos[1]-1 , z_pos[0] ] + 1 )%2
            self.z_corretion_result_on_lattice[ z_pos[1] , z_pos[0]-1 ] = \
                (self.z_corretion_result_on_lattice[ z_pos[1] , z_pos[0]-1 ] + 1 )%2
            self.z_corretion_result_on_lattice[ z_pos[1] , z_pos[0]+1 ] = \
                (self.z_corretion_result_on_lattice[ z_pos[1] , z_pos[0]+1 ] + 1 )%2
    
    def single_X_stabilizer_on_X_result_lattice(self,x_pos):
        assert self.PositionIsXStabilizer(x_pos), \
            "single_X_stabilizer_on_X_result_lattice(): The input is not a X-stabilizer"
        if self.PositionIsHole(x_pos)==False:
            if x_pos[0]!=0 and self.PositionIsHole( (x_pos[0]-1,x_pos[1]) ) == False:
                self.x_corretion_result_on_lattice[ x_pos[1] , x_pos[0]-1 ] = \
                    (self.x_corretion_result_on_lattice[ x_pos[1] , x_pos[0]-1 ] + 1 )%2
            if x_pos[0]!=2*self.length and self.PositionIsHole( (x_pos[0]+1,x_pos[1]) ) == False:
                self.x_corretion_result_on_lattice[ x_pos[1] , x_pos[0]+1 ] = \
                    (self.x_corretion_result_on_lattice[ x_pos[1] , x_pos[0]+1 ] + 1 )%2
            if x_pos[1]!=0 and self.PositionIsHole( (x_pos[0],x_pos[1]-1) ) == False:
                self.x_corretion_result_on_lattice[ x_pos[1]-1 , x_pos[0] ] = \
                    (self.x_corretion_result_on_lattice[ x_pos[1]-1 , x_pos[0] ] + 1 )%2  
            if x_pos[1]!=2*self.width and self.PositionIsHole( (x_pos[0],x_pos[1]+1) ) == False:
                self.x_corretion_result_on_lattice[ x_pos[1]+1 , x_pos[0] ] = \
                    (self.x_corretion_result_on_lattice[ x_pos[1]+1 , x_pos[0] ] + 1 )%2

    def check_correction_result(self):
        # Check Z_correction_result
        for y in range(2*self.width):
            for x in range(2*self.length):
                if self.PositionIsQubit( (x,y) ):
                    if y%2==0 and self.z_corretion_result_on_lattice[y,x]==1:
                        self.single_Z_stabilizer_on_Z_result_lattice( (x,y+1) )
                    if x%2==0 and self.z_corretion_result_on_lattice[y,x]==1:
                        self.single_Z_stabilizer_on_Z_result_lattice( (x+1,y) )
    
        # Check X_correction_result
        for y in range(2*self.width+1):
            for x in range(2*self.length+1):
                if self.PositionIsQubit( (x,y) ):
                    if y%2==0 and self.x_corretion_result_on_lattice[y,x]==1:
                        self.single_X_stabilizer_on_X_result_lattice( (x+1,y) )
                    if x%2==0 and self.x_corretion_result_on_lattice[y,x]==1:
                        self.single_X_stabilizer_on_X_result_lattice( (x,y+1) )
     
    def CorrectionIsCorrect(self):
        
        #print(self.x_corretion_result_on_lattice)
        if np.all( (self.z_corretion_result_on_lattice==0) ) and \
            np.all( (self.x_corretion_result_on_lattice==0) ):
            return True
        else: return False

    def Calculate_total_weight(self):
        tmp = 0
        for y in range(2*self.width+1):
            for x in range(2*self.length+1):
                if self.PositionIsQubit( (x,y) ):
                    if self.x_corretion_on_lattice[y,x]==1 or self.z_corretion_on_lattice[y,x]==1:
                        tmp += 1
        return tmp
    
    def MWPM_decoding(self):
        self.Construct_Z_stabilizer_graph()
        self.Construct_marked_Z_graph()
        if platform.system()=="Windows" : os.system('MWPM -f syndrome_node_graph.txt --minweight > matching_results.txt')
        else : os.system('./MWPM -f syndrome_node_graph.txt --minweight > matching_results.txt')
        self.X_correction()

        self.Construct_X_stabilizer_graph()
        self.Construct_marked_X_graph()
        if platform.system()=="Windows" : os.system('MWPM -f syndrome_node_graph.txt --minweight > matching_results.txt')
        else : os.system('./MWPM -f syndrome_node_graph.txt --minweight > matching_results.txt')
        self.Z_correction()

        print("X correction: \n", np.transpose(np.nonzero(np.transpose(self.x_corretion_on_lattice))))
        print("Z correction: \n", np.transpose(np.nonzero(np.transpose(self.z_corretion_on_lattice))))

    def IRMWPM_decoding(self):
        self.Construct_Z_stabilizer_graph()
        self.Construct_X_stabilizer_graph()
        previous_x_corrections = []
        previous_z_corrections = []

        self.Construct_marked_Z_graph()
        if platform.system()=="Windows" : os.system('MWPM -f syndrome_node_graph.txt --minweight > matching_results.txt')
        else : os.system('./MWPM -f syndrome_node_graph.txt --minweight > matching_results.txt')
        self.X_correction()

        self.Reweight_X_stabilizer_graph()
        self.Construct_marked_X_graph()
        if platform.system()=="Windows" : os.system('MWPM -f syndrome_node_graph.txt --minweight > matching_results.txt')
        else : os.system('./MWPM -f syndrome_node_graph.txt --minweight > matching_results.txt')
        self.Z_correction()

        previous_x_correction = copy.copy(self.x_corretion_on_lattice)
        previous_x_corrections.append(previous_x_correction)    
        self.Reweight_Z_stabilizer_graph()
        self.Construct_marked_Z_graph()
        if platform.system()=="Windows" : os.system('MWPM -f syndrome_node_graph.txt --minweight > matching_results.txt')
        else : os.system('./MWPM -f syndrome_node_graph.txt --minweight > matching_results.txt')
        self.X_correction()

        times_of_reweights = 0
        while not IsSameAsPreviousCorrection(previous_x_corrections, self.x_corretion_on_lattice) and times_of_reweights<=10:
            #print( "Reweight one more time" )
            times_of_reweights += 1
            previous_z_correction = copy.copy(self.z_corretion_on_lattice)
            previous_z_corrections.append(previous_z_correction)
            self.Reweight_X_stabilizer_graph()
            self.Construct_marked_X_graph()
            if platform.system()=="Windows" : os.system('MWPM -f syndrome_node_graph.txt --minweight > matching_results.txt')
            else : os.system('./MWPM -f syndrome_node_graph.txt --minweight > matching_results.txt')
            self.Z_correction()
            
            previous_x_correction = copy.copy(self.x_corretion_on_lattice)
            previous_x_corrections.append(previous_x_correction)
            if not IsSameAsPreviousCorrection(previous_z_corrections, self.z_corretion_on_lattice):
                #print( "Reweight one more time" )
                times_of_reweights += 1
                self.Reweight_Z_stabilizer_graph()
                self.Construct_marked_Z_graph()
                if platform.system()=="Windows" : os.system('MWPM -f syndrome_node_graph.txt --minweight > matching_results.txt')
                else : os.system('./MWPM -f syndrome_node_graph.txt --minweight > matching_results.txt')
                self.X_correction()
        
        print("X correction: \n", np.transpose(np.nonzero(np.transpose(self.x_corretion_on_lattice))))
        print("Z correction: \n", np.transpose(np.nonzero(np.transpose(self.z_corretion_on_lattice))))
        

#                                                                                                        #
#---------------------------------------------------------------------------------------------------------
#                                                                                                        #

###########################################################################
###                     Used for IRMWPM decoding                        ###
def IsSameAsPreviousCorrection (previous_corrections, current_correction):
    for corrcetion in previous_corrections:
        if np.array_equal(corrcetion, current_correction):
            return True
    return False
###                                                                     ###
###########################################################################

class AB_Lattice: # Lattices with Alternative Boundary
    def __init__(self,width,length):
        self.width=width
        self.length=length
        self.lattice = np.zeros((2*width+1,2*length+1))
        self.x_errors_on_lattice = np.zeros((2*width+1,2*length+1))
        self.z_errors_on_lattice = np.zeros((2*width+1,2*length+1))
        self.X_stabilizer_graph = np.ones(( (self.length)*(self.width+1)+2*(self.width+1) ,
                                            (self.length)*(self.width+1)+2*(self.width+1) ))*10000
        self.Z_stabilizer_graph = np.ones(( (self.length+1)*(self.width)+2*(self.length+1) ,
                                            (self.length+1)*(self.width)+2*(self.length+1)  ))*10000
        self.original_X_stabilizer_graph = np.ones(( (self.length)*(self.width+1)+2*(self.width+1) ,
                                            (self.length)*(self.width+1)+2*(self.width+1) ))*10000
        self.original_Z_stabilizer_graph = np.ones(( (self.length+1)*(self.width)+2*(self.length+1) ,
                                            (self.length+1)*(self.width)+2*(self.length+1)  ))*10000
        self.marked_x_stabilzer=[]
        self.marked_z_stabilzer=[]
        self.x_parents=[]
        self.z_parents=[]
        self.X_closest_boundary_nodes=[]
        self.Z_closest_boundary_nodes=[]
        self.x_corretion_on_lattice = np.zeros((2*self.width+1,2*self.length+1))
        self.z_corretion_on_lattice = np.zeros((2*self.width+1,2*self.length+1))
        self.x_corretion_result_on_lattice = np.zeros((2*self.width+1,2*self.length+1))
        self.z_corretion_result_on_lattice = np.zeros((2*self.width+1,2*self.length+1))

    def clear_lattice(self):
        self.lattice = np.zeros((2*self.width+1,2*self.length+1))
        self.x_errors_on_lattice = np.zeros((2*self.width+1,2*self.length+1))
        self.z_errors_on_lattice = np.zeros((2*self.width+1,2*self.length+1))
        self.X_stabilizer_graph = copy.copy(self.original_X_stabilizer_graph)
        self.Z_stabilizer_graph = copy.copy(self.original_Z_stabilizer_graph)
        self.marked_x_stabilzer=[]
        self.marked_z_stabilzer=[]
        self.x_parents=[]
        self.z_parents=[]
        self.X_closest_boundary_nodes=[]
        self.Z_closest_boundary_nodes=[]
        self.x_corretion_on_lattice = np.zeros((2*self.width+1,2*self.length+1))
        self.z_corretion_on_lattice = np.zeros((2*self.width+1,2*self.length+1))
        self.x_corretion_result_on_lattice = np.zeros((2*self.width+1,2*self.length+1))
        self.z_corretion_result_on_lattice = np.zeros((2*self.width+1,2*self.length+1))

    def PositionIsQubit(self,q):
        assert q[0]<=2*self.length and q[0]>=0,"PositionIsQubit(): q[0] is out of range"
        assert q[1]<=2*self.width and q[1]>=0,"PositionIsQubit(): q[1] is out of range"
        if( (q[0]+q[1])%2==0 ): return True;
        else: return False;
        
    def PositionIsZStabilizer(self,q):
        assert q[0]<=2*self.length and q[0]>=0,"PositionIsZStabilizer(): q[0] is out of range"
        assert q[1]<=2*self.width and q[1]>=0,"PositionIsZStabilizer(): q[1] is out of range"
        if( (q[0]+q[1])%2==1 and q[0]%2==0 ): return True;
        else: return False;
    
    def PositionIsXStabilizer(self,q):
        assert q[0]<=2*self.length and q[0]>=0,"PositionIsXStabilizer: q[0] is out of range"
        assert q[1]<=2*self.width and q[1]>=0,"PositionIsXStabilizer: q[1] is out of range"
        if( (q[0]+q[1])%2==1 and q[0]%2==1 ): return True;
        else: return False;
    
    def Receiving_syndrome(self,syndrome_nodes):
        for syndorme_node in syndrome_nodes:
            assert self.PositionIsXStabilizer(syndorme_node) or self.PositionIsZStabilizer(syndorme_node), "Receiving_syndrome(): Invalid syndrome position."
            self.lattice[syndorme_node[1],syndorme_node[0]]=( self.lattice[syndorme_node[1],syndorme_node[0]] + 1 )%2

    # correction-> 0: Nothing, 1: correction, 2: errors, 3: correction + errors
    def show_lattice(self,syndrome=1,correction=1):
        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)
        RED = (0, 0, 255)
        GREEN = (0, 255, 0)
        BLUE = (255, 0, 0)
        assert self.width>0 and self.length>0, "Width and length need to be greater than 0."
        if self.width<=9 or self.length<=9 : interval = 50
        else : interval = int(400/max(self.width, self.length))
        img = np.ones(( (self.width*2)*interval+interval*2, (self.length*2)*interval+interval*2, 3), np.uint8)*255
        
        # Plot the grid
        for y in range(0,self.width+1):
            cv.line(img,(interval,y*2*interval+interval),(self.length*2*interval+interval,y*2*interval+interval),BLACK,3)
        for x in range(0,self.length):       
            cv.line(img,(x*2*interval+2*interval,interval),(x*2*interval+2*interval,self.width*2*interval+interval),BLACK,3)
        
        # Plot the dual lattice
        for y in range(1,self.width+1):
            x=interval
            while x+10<=self.length*2*interval+interval:
                cv.line(img,(x,y*2*interval),(x+10,y*2*interval),BLACK,2)
                x=x+20
        
        for x in range(0,self.length+1):
            y=interval
            while y+10<=self.width*2*interval+interval:
                cv.line(img,(x*2*interval+interval,y),(x*2*interval+interval,y+10),BLACK,2)
                y=y+20
        
        
        # Plot the syndrome
        assert syndrome==0 or syndrome==1, "No such syndrome option."
        if syndrome==1:
            for x in range(0,self.length*2+1):
                for y in range(0,self.width*2+1):
                    if (x+y)%2==1 and self.lattice[y,x]==1:
                        # Z-type error
                        if x%2==1 : cv.circle(img, (x*interval+interval,y*interval+interval), 10, BLUE, -1)
                        # X-type error
                        if x%2==0 : cv.circle(img, (x*interval+interval,y*interval+interval), 10, RED, -1)
        
        # Plot the correction/errors
        # 0: Nothing
        # 1: correction
        # 2: errors
        # 3: correction + errors
        assert correction==0 or correction==1 or correction==2 or correction==3, "No such errors option."
        
        if correction==1:
            for x in range(0,self.length*2+1):
                for y in range(0,self.width*2+1):
                    # X-type correction
                    if (x+y)%2==0 and self.x_corretion_on_lattice[y,x]==1:
                        if x%2==0: cv.line(img,(x*interval+interval,(y-1)*interval+interval) , 
                                               (x*interval+interval,(y+1)*interval+interval), RED, 5)
                        if x%2==1: cv.line(img,((x-1)*interval+interval,y*interval+interval) , 
                                               ((x+1)*interval+interval,y*interval+interval), RED, 5)
                    # Z-type error
                    if (x+y)%2==0 and self.z_corretion_on_lattice[y,x]==1:
                        if x%2==0: cv.line(img,((x-1)*interval+interval,y*interval+interval), 
                                               ((x+1)*interval+interval,y*interval+interval), BLUE, 5)
                        if x%2==1: cv.line(img,(x*interval+interval,(y-1)*interval+interval), 
                                              (x*interval+interval,(y+1)*interval+interval), BLUE, 5)
        
        if correction==2:
            for x in range(0,self.length*2+1):
                for y in range(0,self.width*2+1):
                    # X-type error
                    if (x+y)%2==0 and self.x_errors_on_lattice[y,x]==1:
                        if x%2==0: cv.line(img,(x*interval+interval,(y-1)*interval+interval) , 
                                               (x*interval+interval,(y+1)*interval+interval), RED, 5)
                        if x%2==1: cv.line(img,((x-1)*interval+interval,y*interval+interval) , 
                                               ((x+1)*interval+interval,y*interval+interval), RED, 5)
                    # Z-type error
                    if (x+y)%2==0 and self.z_errors_on_lattice[y,x]==1:
                        if x%2==0: cv.line(img,((x-1)*interval+interval,y*interval+interval), 
                                               ((x+1)*interval+interval,y*interval+interval), BLUE, 5)
                        if x%2==1: cv.line(img,(x*interval+interval,(y-1)*interval+interval), 
                                               (x*interval+interval,(y+1)*interval+interval), BLUE, 5)
        
        if correction==3:
            for x in range(0,self.length*2+1):
                for y in range(0,self.width*2+1):
                    # X-type correction
                    if (x+y)%2==0 and self.x_corretion_result_on_lattice[y,x]==1:
                        if x%2==0: cv.line(img,(x*interval+interval,(y-1)*interval+interval) , 
                                               (x*interval+interval,(y+1)*interval+interval), RED, 5)
                        if x%2==1: cv.line(img,((x-1)*interval+interval,y*interval+interval) , 
                                               ((x+1)*interval+interval,y*interval+interval), RED, 5)
                    # Z-type error
                    if (x+y)%2==0 and self.z_corretion_result_on_lattice[y,x]==1:
                        if x%2==0: cv.line(img,((x-1)*interval+interval,y*interval+interval), 
                                               ((x+1)*interval+interval,y*interval+interval), BLUE, 5)
                        if x%2==1: cv.line(img,(x*interval+interval,(y-1)*interval+interval), 
                                               (x*interval+interval,(y+1)*interval+interval), BLUE, 5)
              
        cv.imshow('Lattice', img) 
        cv.waitKey(0)

    def single_z(self,position):
        assert position[0]<=2*self.length and position[0]>=0,"x-position is out of range"
        assert position[1]<=2*self.width and position[1]>=0,"y-position is out of range"
        assert (position[0]+position[1])%2 != 1 , "The input position is not a qubit"
        self.z_errors_on_lattice[position[1],position[0]] = \
            (self.z_errors_on_lattice[position[1],position[0]]+1)%2
        if position[0]%2 == 1:
            self.lattice[position[1]-1,position[0]]=( self.lattice[position[1]-1,position[0]] + 1 )%2
            self.lattice[position[1]+1,position[0]]=( self.lattice[position[1]+1,position[0]] + 1 )%2
        elif position[1]%2 == 0:
            if position[0]-1>=0: 
                self.lattice[position[1],position[0]-1]=( self.lattice[position[1],position[0]-1] + 1 )%2
            if position[0]+1<=2*self.length:
                self.lattice[position[1],position[0]+1]=( self.lattice[position[1],position[0]+1] + 1 )%2
    
    def single_x(self,position):
        assert position[0]<=2*self.length and position[0]>=0,"x-position is out of range"
        assert position[1]<=2*self.width and position[1]>=0,"y-position is out of range"
        assert (position[0]+position[1])%2 != 1 , "The input position is not a qubit"
        self.x_errors_on_lattice[position[1],position[0]]=(self.x_errors_on_lattice[position[1],position[0]]+1)%2
        if position[0]%2 == 1:
            if position[0]-1 >= 0  : 
                self.lattice[position[1],position[0]-1]=( self.lattice[position[1],position[0]-1] + 1 )%2
            if position[0]+1 <= 2*self.length :
                self.lattice[position[1],position[0]+1]=( self.lattice[position[1],position[0]+1] + 1 )%2
        elif position[1]%2 == 0:
            if position[1]-1 >= 0 : 
                self.lattice[position[1]-1,position[0]]=( self.lattice[position[1]-1,position[0]] + 1 )%2
            if position[1]+1 <= 2*self.width  : 
                self.lattice[position[1]+1,position[0]]=( self.lattice[position[1]+1,position[0]] + 1 )%2
    
    # noise_model: 0=uncorrelated noise model, 1=depolarizing noise model  
    def apply_noises(self,prob=0.1,noise_model=0):
        assert noise_model==0 or noise_model==1, "You've chosen a wrong error model."
        qubits=[]
        NumberOfQubits=0
        for x in range(0,self.length*2+1):
            for y in range(0,self.width*2+1):
                if (x+y)%2==0:
                    qubits.append((x,y))
                    NumberOfQubits += 1

        if noise_model == 0:
            X_error_set = bernoulli(prob).rvs(NumberOfQubits)
            Z_error_set = bernoulli(prob).rvs(NumberOfQubits)
            for e in zip(qubits,X_error_set):
                if e[1]==1: self.single_x(e[0])
            for e in zip(qubits,Z_error_set):
                if e[1]==1: self.single_z(e[0])
        
        elif noise_model==1:
            error_set = bernoulli(prob).rvs(NumberOfQubits)
            for e in zip(qubits,error_set):
                if e[1]==1:
                    temp = random.randint(1,3)
                    if temp==1: self.single_x(e[0])
                    elif temp==2: self.single_z(e[0])
                    elif temp==3:
                        self.single_x(e[0])
                        self.single_z(e[0])

    def Construct_X_stabilizer_graph(self):
        Number_X_Stabilizer = (self.length)*(self.width+1)  
        self.X_stabilizer_graph = np.ones(  (Number_X_Stabilizer+2*(self.width+1),Number_X_Stabilizer+2*(self.width+1))  )*10000
        np.fill_diagonal(self.X_stabilizer_graph,0)
        for x in range(0,self.length):
            for y in range(0,self.width+1):
                if x!=self.length-1 :
                    self.X_stabilizer_graph[ y*(self.length)+x+1,  y*(self.length)+x  ] = 2
                    self.X_stabilizer_graph[ y*(self.length)+x, y*(self.length)+x+1 ] = 2
                if y!=self.width :
                    self.X_stabilizer_graph[ (y+1)*(self.length)+x,  y*(self.length)+x  ] = 2
                    self.X_stabilizer_graph[ y*(self.length)+x, (y+1)*(self.length)+x ] = 2

        for y in range(0,self.width+1):
            self.X_stabilizer_graph[ y*self.length,  y+Number_X_Stabilizer  ] = 2
            self.X_stabilizer_graph[ y+Number_X_Stabilizer, y*self.length   ] = 2
            self.X_stabilizer_graph[ y*self.length+self.length-1,  y+Number_X_Stabilizer+self.width+1  ] = 2
            self.X_stabilizer_graph[ y+Number_X_Stabilizer+self.width+1,  y*self.length+self.length-1   ] = 2

        np.set_printoptions(suppress=True)
        self.original_X_stabilizer_graph = copy.copy(self.X_stabilizer_graph)
    
    def Construct_Z_stabilizer_graph(self,prob=0.1):
        Number_Z_Stabilizer = (self.length+1)*(self.width) 
        self.Z_stabilizer_graph = np.ones(  ( Number_Z_Stabilizer+2*(self.length+1),Number_Z_Stabilizer+2*(self.length+1))  )*10000
        np.fill_diagonal(self.Z_stabilizer_graph,0)
        
        for x in range(0,self.length+1):
            for y in range(0,self.width):
                if(x!=self.length):
                    self.Z_stabilizer_graph[ y*(self.length+1)+x+1,  y*(self.length+1)+x  ] = 2
                    self.Z_stabilizer_graph[ y*(self.length+1)+x, y*(self.length+1)+x+1 ] = 2
                if(y!=self.width-1):
                    self.Z_stabilizer_graph[ (y+1)*(self.length+1)+x,  y*(self.length+1)+x  ] = 2
                    self.Z_stabilizer_graph[ y*(self.length+1)+x, (y+1)*(self.length+1)+x ] = 2
        
        for x in range(0,self.length+1):
            self.Z_stabilizer_graph[ x,  x+Number_Z_Stabilizer  ] = 2
            self.Z_stabilizer_graph[ x+Number_Z_Stabilizer,  x  ] = 2
            self.Z_stabilizer_graph[ x+(self.length+1)*(self.width-1), x+Number_Z_Stabilizer+self.length+1 ] = 2
            self.Z_stabilizer_graph[ x+Number_Z_Stabilizer+self.length+1, x+(self.length+1)*(self.width-1) ] = 2
        
        np.set_printoptions(suppress=True)
        self.original_Z_stabilizer_graph = copy.copy(self.Z_stabilizer_graph)
        #np.savetxt("Z_stabilizer_graph.csv",self.Z_stabilizer_graph, delimiter=',', fmt='%f')
    
    def Construct_marked_X_graph(self):
        #######################################################################
        ############# Dijkstra's Algorithm on X_stabilizer_graph ##############
        #######################################################################
        Number_X_Stabilizer = (self.length)*(self.width+1)
        self.marked_x_stabilzer=[]
        for m in range(Number_X_Stabilizer):
            if self.lattice[2*( int( m/(self.length) ) ), (m%self.length)*2+1  ]==1:
                self.marked_x_stabilzer.append(m)
        f=open("syndrome_node_graph.txt","w")
        f.write( "%d\n" % (len(self.marked_x_stabilzer)*2) ) 
        num_edges = len(self.marked_x_stabilzer)*(len(self.marked_x_stabilzer)-1)+len(self.marked_x_stabilzer)
        f.write( "%d\n" %  num_edges )
        self.x_parents=[]
        self.X_closest_boundary_nodes=[]
        for d in range(len(self.marked_x_stabilzer)):
            src = self.marked_x_stabilzer[d]
            row = len(self.X_stabilizer_graph)
            col = len(self.X_stabilizer_graph[0])
            dist = [float("Inf")] * row
            parent = [-1] * row
            dist[src] = 0
            # Add all vertices in queue
            queue = []
            for j in range(row):
                queue.append(j)

            while queue:
                u = minDistance(dist,queue) 
                queue.remove(u)
                for i in range(col):
                    if self.X_stabilizer_graph[u][i] and i in queue:
                        if dist[u] + self.X_stabilizer_graph[u][i] < dist[i]:
                            dist[i] = dist[u] + self.X_stabilizer_graph[u][i]
                            parent[i] = u

            self.x_parents.append(parent)
            # print the constructed distance array
            for s in range(d+1, len(self.marked_x_stabilzer)):
                f.write("%d %d %f\n" % (d, s, dist[self.marked_x_stabilzer[s]]))
                #printPath(parent,marked_x_stabilzer[s])
            
            closest_boundary_node=-1
            distance_to_boundary = 10000

            for b in range(Number_X_Stabilizer, Number_X_Stabilizer+2*(self.width+1) ):
                if dist[b] < distance_to_boundary:
                    distance_to_boundary = dist[b]
                    closest_boundary_node = b

            self.X_closest_boundary_nodes.append(closest_boundary_node)                             
            f.write("%d %d %f\n" % (d, len(self.marked_x_stabilzer)+d, distance_to_boundary))
            
        for b1 in range(len(self.marked_x_stabilzer),len(self.marked_x_stabilzer)*2):
            for b2 in range(b1+1,len(self.marked_x_stabilzer)*2):
                f.write("%d %d %f\n" % (b1, b2, 0))

    def Construct_marked_Z_graph(self,prob=0.1):
        #######################################################################
        ############# Dijkstra's Algorithm on Z_stabilizer_graph ##############
        #######################################################################
        Number_Z_Stabilizer = (self.length+1)*(self.width)
        self.marked_z_stabilzer=[]
        for m in range(Number_Z_Stabilizer):
            if self.lattice[ 2*( int( m/(self.length+1) ) )+1 , 2*( int( m%(self.length+1) ) ) ]==1:
                self.marked_z_stabilzer.append(m)
        f=open("syndrome_node_graph.txt","w")
        f.write( "%d\n" % (len(self.marked_z_stabilzer)*2) )
        num_edges = len(self.marked_z_stabilzer)*(len(self.marked_z_stabilzer)-1)+len(self.marked_z_stabilzer)
        f.write( "%d\n" %  num_edges ) 
        self.z_parents=[]
        self.Z_closest_boundary_nodes=[]
        for d in range(len(self.marked_z_stabilzer)):
            src = self.marked_z_stabilzer[d]
            row = len(self.Z_stabilizer_graph)
            col = len(self.Z_stabilizer_graph[0])
            dist = [float("Inf")] * row
            parent = [-1] * row
            dist[src] = 0
            # Add all vertices in queue
            queue = []
            for j in range(row):
                queue.append(j)

            while queue:
                u = minDistance(dist,queue) 
                queue.remove(u)
                for i in range(col):
                    if self.Z_stabilizer_graph[u][i] and i in queue:
                        if dist[u] + self.Z_stabilizer_graph[u][i] < dist[i]:
                            dist[i] = dist[u] + self.Z_stabilizer_graph[u][i]
                            parent[i] = u

            self.z_parents.append(parent)
            # print the constructed distance array
            for s in range(d+1, len(self.marked_z_stabilzer)):
                f.write("%d %d %f\n" % (d, s, dist[self.marked_z_stabilzer[s]]))
                #printPath(parent,marked_x_stabilzer[s])
            
            closest_boundary_node=-1
            distance_to_boundary = 10000
            
            for b in range(Number_Z_Stabilizer, Number_Z_Stabilizer+2*(self.length+1) ):
                if dist[b] < distance_to_boundary:
                    distance_to_boundary = dist[b]
                    closest_boundary_node = b
            self.Z_closest_boundary_nodes.append(closest_boundary_node)                             
            f.write("%d %d %f\n" % (d, len(self.marked_z_stabilzer)+d, distance_to_boundary))
            
        for b1 in range(len(self.marked_z_stabilzer),len(self.marked_z_stabilzer)*2):
            for b2 in range(b1+1,len(self.marked_z_stabilzer)*2):
                f.write("%d %d %f\n" % (b1, b2, 0))

    def string_Z_correction(self,parent,src,dst):
        Number_X_Stabilizer = (self.length)*(self.width+1)
        current_mark=dst
        if current_mark < Number_X_Stabilizer:
            y_c = 2*( int( current_mark/(self.length) ) )
            x_c = 2*( current_mark%self.length ) + 1
        elif current_mark < Number_X_Stabilizer + self.width+1:
            y_c = 2*( current_mark - Number_X_Stabilizer  )
            x_c = -1    
        else:
            y_c = 2*( current_mark - (Number_X_Stabilizer+self.width+1) )
            x_c = 2*self.length+1
        while current_mark!=src:
            next_mark=parent[current_mark]
            y_n = 2*( int( next_mark/(self.length) ) )
            x_n = 2*( int( next_mark%(self.length) ) ) + 1
            assert y_c==y_n or x_c==x_n, "string_Z_correction(): The edge between the two marks doesn't exist."
            assert y_c!=y_n or x_c!=x_n, "string_Z_correction(): The two marks are the same."
            if y_c==y_n:
                assert x_c==x_n+2 or x_c==x_n-2, "string_Z_correction(): The two marks are not next two each other."
                self.z_corretion_on_lattice[y_c,int( (x_c+x_n)/2 )]=1
            elif x_c==x_n:
                assert y_c==y_n+2 or y_c==y_n-2, "string_correction(): The two marks are not next two each other."
                self.z_corretion_on_lattice[int( (y_c+y_n)/2 ), x_c]=1
            current_mark=next_mark
            y_c=y_n
            x_c=x_n
            
    def string_X_correction(self,parent,src,dst):
        Number_Z_Stabilizer = (self.length+1)*(self.width)
        current_mark=dst
        if current_mark < Number_Z_Stabilizer:
            y_c = 2*( int( current_mark/(self.length+1) ) ) + 1
            x_c = 2*( current_mark%(self.length+1) )
        elif current_mark < Number_Z_Stabilizer+self.length+1:
            y_c = -1
            x_c = 2*( current_mark - Number_Z_Stabilizer ) 
        else:
            y_c = 2*self.width+1
            x_c = 2*( current_mark - (Number_Z_Stabilizer+self.length+1) )
        while current_mark!=src:
            next_mark=parent[current_mark]
            y_n = 2*( int( next_mark/(self.length+1) ) ) + 1
            x_n = 2*(  next_mark%(self.length+1)  ) 
            #if not (y_c==y_n or x_c==x_n) :
            #  print(dst,current_mark,next_mark,src)
            #  print(x_c,y_c,x_n,y_n)
            #if not (y_c!=y_n or x_c!=x_n):
            #  print(dst,current_mark,next_mark,src)
            #  print(x_c,y_c,x_n,y_n)
            assert y_c==y_n or x_c==x_n, "string_X_correction(): The edge between the two marks doesn't exist."
            assert y_c!=y_n or x_c!=x_n, "string_X_correction(): The two marks are the same."
            if y_c==y_n:
                assert x_c==x_n+2 or x_c==x_n-2, "string_X_correction(): The two marks are not next two each other."
                self.x_corretion_on_lattice[y_c,int( (x_c+x_n)/2 )]=1
            elif x_c==x_n:
                assert y_c==y_n+2 or y_c==y_n-2, "string_X_correction(): The two marks are not next two each other."
                self.x_corretion_on_lattice[int( (y_c+y_n)/2 ), x_c]=1
            current_mark=next_mark
            y_c=y_n
            x_c=x_n

    def Z_correction(self):
        correction_pairs = []
        self.z_corretion_on_lattice = np.zeros((2*self.width+1,2*self.length+1))
        self.z_corretion_result_on_lattice = np.zeros((2*self.width+1,2*self.length+1))
        with open("matching_results.txt", "r", encoding = "utf-8") as fp:
            for i in fp.readlines():
                tmp = i.split(" ")
                correction_pairs.append([int(tmp[0]), int(tmp[1])])
        for p in correction_pairs:
            assert p[0] < len(self.marked_x_stabilzer)*2 and p[1] < len(self.marked_x_stabilzer)*2, \
                "Z_correction: The input pair is wrong"
            if p[0] < len(self.marked_x_stabilzer):
                if p[1] < len(self.marked_x_stabilzer):
                    self.string_Z_correction(self.x_parents[p[0]],self.marked_x_stabilzer[p[0]],self.marked_x_stabilzer[p[1]])
                else:
                    self.string_Z_correction(self.x_parents[p[0]], 
                                             self.marked_x_stabilzer[p[0]] , 
                                             self.X_closest_boundary_nodes[p[1]-len(self.marked_x_stabilzer)])
        self.z_corretion_result_on_lattice = self.z_errors_on_lattice + self.z_corretion_on_lattice
        self.z_corretion_result_on_lattice = modulo(self.z_corretion_result_on_lattice,2)
    
    def X_correction(self):
        correction_pairs = []
        self.x_corretion_on_lattice = np.zeros((2*self.width+1,2*self.length+1))
        self.x_corretion_result_on_lattice = np.zeros((2*self.width+1,2*self.length+1))
        with open("matching_results.txt", "r", encoding = "utf-8") as fp:
            for i in fp.readlines():
                tmp = i.split(" ")
                correction_pairs.append([int(tmp[0]), int(tmp[1])])
        for p in correction_pairs:
            assert p[0] < len(self.marked_z_stabilzer)*2 and p[1] < len(self.marked_z_stabilzer)*2, \
                "X_correction: The input pair is wrong"
            if p[0] < len(self.marked_z_stabilzer):
                if p[1] < len(self.marked_z_stabilzer):
                    self.string_X_correction(self.z_parents[p[0]],self.marked_z_stabilzer[p[0]],self.marked_z_stabilzer[p[1]])
                else:
                    self.string_X_correction(self.z_parents[p[0]], 
                                             self.marked_z_stabilzer[p[0]] , 
                                             self.Z_closest_boundary_nodes[p[1]-len(self.marked_z_stabilzer)])
        self.x_corretion_result_on_lattice = self.x_errors_on_lattice + self.x_corretion_on_lattice
        self.x_corretion_result_on_lattice = modulo(self.x_corretion_result_on_lattice,2)

    def Reweight_X_stabilizer_graph(self):
        Number_X_Stabilizer = (self.length)*(self.width+1)
        self.X_stabilizer_graph = copy.copy(self.original_X_stabilizer_graph)
        for x in range(0,self.length):
            for y in range(0,self.width+1): # (2*x+1, 2*y)
                if x!=self.length-1 and self.x_corretion_on_lattice[ y*2 , x*2+1+1 ] == 1:
                    self.X_stabilizer_graph[ y*(self.length)+x+1,  y*(self.length)+x  ] = 0.01 
                    self.X_stabilizer_graph[ y*(self.length)+x, y*(self.length)+x+1 ] = 0.01
                if(y!=self.width) and self.x_corretion_on_lattice[ y*2+1 , x*2+1 ] == 1:
                    self.X_stabilizer_graph[ (y+1)*(self.length)+x,  y*(self.length)+x  ] = 0.01
                    self.X_stabilizer_graph[ y*(self.length)+x, (y+1)*(self.length)+x ] = 0.01

        for y in range(0,self.width+1):
            if self.x_corretion_on_lattice[ 2*y , 0]==1:
                self.X_stabilizer_graph[ y*self.length,  y+Number_X_Stabilizer  ] = 0.01
                self.X_stabilizer_graph[ y+Number_X_Stabilizer, y*self.length   ] = 0.01
            if self.x_corretion_on_lattice[ 2*y , 2*self.length]==1:
                self.X_stabilizer_graph[ y*self.length+self.length-1,  
                                    y+Number_X_Stabilizer+self.width+1  ] = 0.01
                self.X_stabilizer_graph[ y+Number_X_Stabilizer+self.width+1, 
                                    y*self.length+self.length-1  ] = 0.01

    def Reweight_Z_stabilizer_graph(self):
        Number_Z_Stabilizer = (self.length+1)*(self.width)
        self.Z_stabilizer_graph = copy.copy(self.original_Z_stabilizer_graph)
        for x in range(0,self.length+1):
            for y in range(0,self.width): # (2*x, 2*y+1)
                if (x!=self.length) and self.z_corretion_on_lattice[ 2*y+1 , 2*x+1]==1 :
                    self.Z_stabilizer_graph[ y*(self.length+1)+x+1,  y*(self.length+1)+x  ] = 0.01
                    self.Z_stabilizer_graph[ y*(self.length+1)+x, y*(self.length+1)+x+1 ] = 0.01
                if(y!=self.width-1) and self.z_corretion_on_lattice[ 2*y+1+1 , 2*x]==1 :
                    self.Z_stabilizer_graph[ (y+1)*(self.length+1)+x,  y*(self.length+1)+x  ] = 0.01
                    self.Z_stabilizer_graph[ y*(self.length+1)+x, (y+1)*(self.length+1)+x ] = 0.01
        
        for x in range(0,self.length+1):
            if self.z_corretion_on_lattice[ 0 , 2*x]==1:
                self.Z_stabilizer_graph[ x,  x+Number_Z_Stabilizer  ] = 0.01
                self.Z_stabilizer_graph[ x+Number_Z_Stabilizer,  x  ] = 0.01
            if self.z_corretion_on_lattice[  2*self.width , 2*x]==1:
                self.Z_stabilizer_graph[ x+(self.length+1)*(self.width-1),
                                        x+Number_Z_Stabilizer+self.length+1 ] = 0.01
                self.Z_stabilizer_graph[ x+Number_Z_Stabilizer+self.length+1, 
                                        x+(self.length+1)*(self.width-1)  ] = 0.01

    def single_Z_stabilizer_on_Z_result_lattice(self,z_pos):
        assert self.PositionIsZStabilizer(z_pos), \
            "single_Z_stabilizer_on_Z_result_lattice(): The input is not a Z-stabilizer"
        if z_pos[1]+1 <= 2*self.width:
            self.z_corretion_result_on_lattice[ z_pos[1]+1 , z_pos[0] ] = \
                (self.z_corretion_result_on_lattice[ z_pos[1]+1 , z_pos[0] ] + 1 )%2
        if z_pos[1]-1 >= 0:
            self.z_corretion_result_on_lattice[ z_pos[1]-1 , z_pos[0] ] = \
                (self.z_corretion_result_on_lattice[ z_pos[1]-1 , z_pos[0] ] + 1 )%2
        if z_pos[0]-1 >= 0:
            self.z_corretion_result_on_lattice[ z_pos[1] , z_pos[0]-1 ] = \
                (self.z_corretion_result_on_lattice[ z_pos[1] , z_pos[0]-1 ] + 1 )%2
        if z_pos[0]+1 <= 2*self.length:
            self.z_corretion_result_on_lattice[ z_pos[1] , z_pos[0]+1 ] = \
                (self.z_corretion_result_on_lattice[ z_pos[1] , z_pos[0]+1 ] + 1 )%2
    
    def single_X_stabilizer_on_X_result_lattice(self,x_pos):
        assert self.PositionIsXStabilizer(x_pos), \
            "single_X_stabilizer_on_X_result_lattice(): The input is not a X-stabilizer"
        if x_pos[0]!=0 :
            self.x_corretion_result_on_lattice[ x_pos[1] , x_pos[0]-1 ] = \
                (self.x_corretion_result_on_lattice[ x_pos[1] , x_pos[0]-1 ] + 1 )%2
        if x_pos[0]!=2*self.length :
            self.x_corretion_result_on_lattice[ x_pos[1] , x_pos[0]+1 ] = \
                (self.x_corretion_result_on_lattice[ x_pos[1] , x_pos[0]+1 ] + 1 )%2
        if x_pos[1]!=0 :
            self.x_corretion_result_on_lattice[ x_pos[1]-1 , x_pos[0] ] = \
                (self.x_corretion_result_on_lattice[ x_pos[1]-1 , x_pos[0] ] + 1 )%2  
        if x_pos[1]!=2*self.width :
            self.x_corretion_result_on_lattice[ x_pos[1]+1 , x_pos[0] ] = \
                (self.x_corretion_result_on_lattice[ x_pos[1]+1 , x_pos[0] ] + 1 )%2

    def check_correction_result(self):
        # Check Z_correction_result
        for y in range(2*self.width):
            for x in range(2*self.length+1):
                if self.PositionIsQubit( (x,y) ):
                    if y%2==0 and self.z_corretion_result_on_lattice[y,x]==1:
                        self.single_Z_stabilizer_on_Z_result_lattice( (x,y+1) )
                        #self.show_lattice(1,3)
                    if x%2==0 and self.z_corretion_result_on_lattice[y,x]==1:
                        self.single_Z_stabilizer_on_Z_result_lattice( (x+1,y) )
                        #self.show_lattice(1,3)
                
        
        # Check X_correction_result
        for y in range(2*self.width+1):
            for x in range(2*self.length):
                if self.PositionIsQubit( (x,y) ):
                    if y%2==0 and self.x_corretion_result_on_lattice[y,x]==1:
                        self.single_X_stabilizer_on_X_result_lattice( (x+1,y) )
                        #self.show_lattice(1,3)
                    if x%2==0 and self.x_corretion_result_on_lattice[y,x]==1:
                        self.single_X_stabilizer_on_X_result_lattice( (x,y+1) )
                        #self.show_lattice(1,3)

    def CorrectionIsCorrect(self):
        #print(self.x_corretion_result_on_lattice)
        if np.all( (self.z_corretion_result_on_lattice==0) ) and \
            np.all( (self.x_corretion_result_on_lattice==0) ):
            return True
        else: return False
    
    def Calculate_total_weight(self):
        tmp = 0
        for y in range(2*self.width+1):
            for x in range(2*self.length+1):
                if self.PositionIsQubit( (x,y) ):
                    if self.x_corretion_on_lattice[y,x]==1 or self.z_corretion_on_lattice[y,x]==1:
                        tmp += 1
        return tmp

    def MWPM_decoding(self):
        self.Construct_Z_stabilizer_graph()
        self.Construct_marked_Z_graph()
        if platform.system()=="Windows" : os.system('MWPM -f syndrome_node_graph.txt --minweight > matching_results.txt')
        else : os.system('./MWPM -f syndrome_node_graph.txt --minweight > matching_results.txt')
        self.X_correction()

        self.Construct_X_stabilizer_graph()
        self.Construct_marked_X_graph()
        if platform.system()=="Windows" : os.system('MWPM -f syndrome_node_graph.txt --minweight > matching_results.txt')
        else : os.system('./MWPM -f syndrome_node_graph.txt --minweight > matching_results.txt')
        self.Z_correction()

        print("X correction: \n", np.transpose(np.nonzero(np.transpose(self.x_corretion_on_lattice))))
        print("Z correction: \n", np.transpose(np.nonzero(np.transpose(self.z_corretion_on_lattice))))
    
    def IRMWPM_decoding(self):
        self.Construct_Z_stabilizer_graph()
        self.Construct_X_stabilizer_graph()
        previous_x_corrections = []
        previous_z_corrections = []

        self.Construct_marked_Z_graph()
        if platform.system()=="Windows" : os.system('MWPM -f syndrome_node_graph.txt --minweight > matching_results.txt')
        else : os.system('./MWPM -f syndrome_node_graph.txt --minweight > matching_results.txt')
        self.X_correction()

        self.Reweight_X_stabilizer_graph()
        self.Construct_marked_X_graph()
        if platform.system()=="Windows" : os.system('MWPM -f syndrome_node_graph.txt --minweight > matching_results.txt')
        else : os.system('./MWPM -f syndrome_node_graph.txt --minweight > matching_results.txt')
        self.Z_correction()

        previous_x_correction = copy.copy(self.x_corretion_on_lattice)
        previous_x_corrections.append(previous_x_correction)    
        self.Reweight_Z_stabilizer_graph()
        self.Construct_marked_Z_graph()
        if platform.system()=="Windows" : os.system('MWPM -f syndrome_node_graph.txt --minweight > matching_results.txt')
        else : os.system('./MWPM -f syndrome_node_graph.txt --minweight > matching_results.txt')
        self.X_correction()

        times_of_reweights = 0
        while not IsSameAsPreviousCorrection(previous_x_corrections, self.x_corretion_on_lattice) and times_of_reweights<=10:
            #print( "Reweight one more time" )
            times_of_reweights += 1
            previous_z_correction = copy.copy(self.z_corretion_on_lattice)
            previous_z_corrections.append(previous_z_correction)
            self.Reweight_X_stabilizer_graph()
            self.Construct_marked_X_graph()
            if platform.system()=="Windows" : os.system('MWPM -f syndrome_node_graph.txt --minweight > matching_results.txt')
            else : os.system('./MWPM -f syndrome_node_graph.txt --minweight > matching_results.txt')
            self.Z_correction()
            
            previous_x_correction = copy.copy(self.x_corretion_on_lattice)
            previous_x_corrections.append(previous_x_correction)
            if not IsSameAsPreviousCorrection(previous_z_corrections, self.z_corretion_on_lattice):
                #print( "Reweight one more time" )
                times_of_reweights += 1
                self.Reweight_Z_stabilizer_graph()
                self.Construct_marked_Z_graph()
                if platform.system()=="Windows" : os.system('MWPM -f syndrome_node_graph.txt --minweight > matching_results.txt')
                else : os.system('./MWPM -f syndrome_node_graph.txt --minweight > matching_results.txt')
                self.X_correction()

        print("X correction: \n", np.transpose(np.nonzero(np.transpose(self.x_corretion_on_lattice))))
        print("Z correction: \n", np.transpose(np.nonzero(np.transpose(self.z_corretion_on_lattice))))