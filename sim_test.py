import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from variables import *
####### Variables ##########

#for i in blocks_per_user :
#    print(i)
#print(blocks_per_user)
#print(tau)
# This function calculates the wheighted volume for all users transactions seen from the perspective of a certain block "block_number"
def wheigted_volumes(blocks_per_user,block_number,tau):
    # x is a 0 matrix used to facilitate the calculations with Numpy
    x=np.zeros((len(blocks_per_user),20))
    # W is list of the wheighted volumes for all transactions
    w=[]
    
# this loop fills 
    for i in range(len(blocks_per_user)):
        x[i,:len(blocks_per_user[i])]=blocks_per_user[i]
        y=np.around(vij*np.exp((x-block_number)/tau),2)

    for i in range(len(y)):
        w.append(list(y[i,:len(blocks_per_user[i])]))
    for i in range(len(w)):
        # this is only to say that when the block has not happened yet, the volume is zero.
        for j in range(len(w[i])):
            if w[i][j]>vij:
                w[i][j]=0
    return(w)

    
#print(wheigted_volumes(blocks_per_user,60,tau))
# calculates total wheighted volume for all users, and by user, seen from a block "block".
def cum_vol(L,block,tau):
    x=wheigted_volumes(L,block,tau)
    for i in range(len(x)):
        x[i]=sum(x[i])
    return(np.around(x,2))

#print(wheigted_volumes(blocks_per_user,30,tau))
#print(cum_vol(blocks_per_user,30,tau))
#print(tau)
# The list of block used in the simulation, we chose 100 blocks for the demonstration.
block = list(range(1,b+1))
#print(block)
# Calculates the total wheighted volume for each user for the 100 blocks.
def total_wheighted_volume_byuser(user_blocks,block,tau):
    result=[]
    result = (map(lambda block:list(cum_vol(user_blocks,block,tau)), block))
    wheighted=list(result)
    return(wheighted)

# calculates the total wheihted volume for by all users for 100 blocks.
def total_volume_generated_byblock(total_wheighted_volume_byuser):
    x=[sum(i) for i in total_wheighted_volume_byuser]
    x=np.array(x)
    total_volume_per_block=np.around(x,2)
    return(total_volume_per_block)

#Testing above functions.
TU=total_wheighted_volume_byuser(blocks_per_user,block,tau)
#print(TU)

T=total_volume_generated_byblock(TU)
#print(T)
# Unzipping the wheighted volume list and making one list per user. The list contains the total wheighted volume a user has at all 100 blocks

user1=(list(list(zip(*TU))[0]))
user2=(list(list(zip(*TU))[1]))
user3=(list(list(zip(*TU))[2]))
user4=(list(list(zip(*TU))[3]))
user5=(list(list(zip(*TU))[4]))
# let's note WV as the weighted volume
#Plotting the weighted volume evolution per user

f1 = plt.figure(1)

plt.style.use('ggplot')
plt.plot(block, user1, label = "user1 WV by block")
plt.plot(block, user2, label = "user2 WV volume by block")
plt.plot(block, user3, label = "user3 WV volume by block")
plt.plot(block, user4, label = "user4 WV volume by block")
plt.plot(block, user5, label = "user5 WV volume by block")
plt.plot(block, T, label = "total WV in the platform")

plt.xlabel("block number")
plt.ylabel("Eth")

plt.title('Users WV as time passes')
# show a legend on the plot
plt.legend()
 
# function to show the plot


# Transforming T and TU into arrays to allow scalar product.
TU=np.array(total_wheighted_volume_byuser(blocks_per_user,block,tau))
T=np.array(total_volume_generated_byblock(TU)).reshape(-1,1)

#print(TU,T)
# calculating pool share
pool_share=[]
pool_share=(TU/T)
#print(pool_share)

#Let's refer to pool share percentage with PSP 
#And refer to pool share with PS
# unzipping the PS for each user for all blocks
ps_user1=(list(list(zip(*pool_share))[0]))
ps_user2=(list(list(zip(*pool_share))[1]))
ps_user3=(list(list(zip(*pool_share))[2]))
ps_user4=(list(list(zip(*pool_share))[3]))
ps_user5=(list(list(zip(*pool_share))[4]))

# plotting the pool share
'''f1 = plt.figure(2)
plt.plot(block, ps_user1, label = "user1 PSP by block")
plt.plot(block, ps_user2, label = "user2 PSP by block")
plt.plot(block, ps_user3, label = "user3 PSP by block")
plt.plot(block, ps_user4, label = "user4 PSP by block")
plt.plot(block, ps_user5, label = "user5 PSP by block")'''

plt.xlabel("block number")
plt.ylabel("PSP in %")


plt.title('PSPs as time passes')
# show a legend on the plot
plt.legend()
 
# function to show the plot


#Finding how many sales were made in a block
def volume_per_block(Lu,b):
    flat_lo = [item for sublist in Lu for item in sublist]
    Volume=flat_lo.count(b)*vij
    return Volume

#Grouping all block rewards in one matrix. Each element of the matrix represents the block rewards every user have received
block_rewards=[]
for i in range(100):         
    block_rewards.append(list(np.nan_to_num(np.around(pool_share[i]*volume_per_block(blocks_per_user,i)*f,2))))

# Cumulating the balances for all 100 blocks. For block number 5 for example, the balance will be a sum of all blocks before in addition to block 5.
balance_user1=np.cumsum(np.array((list(list(zip(*block_rewards))[0]))))
balance_user2=np.cumsum(np.array((list(list(zip(*block_rewards))[1]))))
balance_user3=np.cumsum(np.array((list(list(zip(*block_rewards))[2]))))
balance_user4=np.cumsum(np.array((list(list(zip(*block_rewards))[3]))))
balance_user5=np.cumsum(np.array((list(list(zip(*block_rewards))[4]))))

total_balance=balance_user1+balance_user2+balance_user3+balance_user4+balance_user5

# plotting balances of each user
f1 = plt.figure(3)

plt.style.use('seaborn')
plt.plot(block, balance_user1, label = "user1 balance by block number")
plt.plot(block, balance_user2, label = "user2 balance by block number")
plt.plot(block, balance_user3, label = "user3 balance by block number")
plt.plot(block, balance_user4, label = "user4 balance by block number")
plt.plot(block, balance_user5, label = "user5 balance by block number")
plt.plot(block, total_balance, label = "Total balance by block number")

plt.title('Users balance as time passes')
# show a legend on the plot
plt.legend()
# function to show the plot
plt.show()