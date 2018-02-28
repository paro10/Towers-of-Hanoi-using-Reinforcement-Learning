'''TOH_VI.py

Value Iteration for Markov Decision Processes.
'''
import math
# Edit the returned name to ensure you get credit for the assignment.
def student_name():
   return "Banerjee, Paromita" # For an autograder.

Vkplus1 = {}
Q_Values_Dict = {}

def one_step_of_VI(S, A, T, R, gamma, Vk):
   '''S is list of all the states defined for this MDP.
   A is a list of all the possible actions.
   T is a function representing the MDP's transition model.
   R is a function representing the MDP's reward function.
   gamma is the discount factor.
   The current value of each state s is accessible as Vk[s].
   '''

   global Q_Values_Dict, Vkplus1

   delta_max=0 #setting initial value to 0

   for s in S:
      v_list=[] #list to hold values on every iteration
      for a in A:
         v=0 #initialize to 0
         for sp in S: #For each state and action, check neighbouring state
            probability=T(s,a,sp) #calculate probability
            reward=R(s,a,sp) #calculate reward for given s,a and sp
            v+=probability*(reward+gamma*Vk[sp]) #update value
         Q_Values_Dict[(s,a)]=v #append to Q_Values_Dict
         v_list.append(v)

      if len(v_list)>0: #if list not empty
         Vkplus1[s]=max(v_list) #Select max element

      delta_max=max(delta_max,abs(Vkplus1[s]-Vk[s])) #update delta_max values

   return (Vkplus1, delta_max)
   #return (Vk, 0) # placeholder

def return_Q_values(S, A):
   '''Return the dictionary whose keys are (state, action) tuples,
   and whose values are floats representing the Q values from the
   most recent call to one_step_of_VI. This is the normal case, and
   the values of S and A passed in here can be ignored.
   However, if no such call has been made yet, use S and A to
   create the answer dictionary, and use 0.0 for all the values.
   '''

   for s in S:
      for a in A: #loop over all state-action pairs
         if (s,a) not in Q_Values_Dict: #check if key-value pair not present
            Q_Values_Dict[(s,a)]=0.0 #update value

   return Q_Values_Dict # placeholder

Policy = {}
def extract_policy(S, A):
   '''Return a dictionary mapping states to actions. Obtain the policy
   using the q-values most recently computed.  If none have yet been
   computed, call return_Q_values to initialize q-values, and then
   extract a policy.  Ties between actions having the same (s, a) value
   can be broken arbitrarily.
   '''
   global Policy, Q_Values_Dict
   Policy = {}

   # Add code here
   if not Q_Values_Dict: #check if empty
      Q_Values_Dict = return_Q_values(S, A)

   for s in S:
      v_list = [] #empty list to append Q-values
      for a in A:
         v_list.append(Q_Values_Dict[(s, a)])
      idx=v_list.index(max(v_list)) #extract index of max element in list
      Policy[s] = A[idx] #for each state, find action with max Q-value and assign to Policy

   print("Policy",Policy)

   return Policy

def apply_policy(s):
   '''Return the action that your current best policy implies for state s.'''
   global Policy
   #return None # placeholder
   return Policy[s]


