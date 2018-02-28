'''paro10b_Q_Learn.py


This is part of the UW Intro to AI Starter Code for Reinforcement Learning.

'''
import random
# Edit the returned name to ensure you get credit for the assignment.
def student_name():
#*** ADD OR CHANGE CODE HERE ***
   return "Banerjee, Paromita" # For an autograder.

STATES=None; ACTIONS=None; UQV_callback=None; Q_VALUES=None,
is_valid_goal_state=None; Terminal_state = None
USE_EXPLORATION_FUNCTION = None
INITIAL_STATE = None
def setup(states, actions, q_vals_dict, update_q_value_callback,\
    goal_test, terminal, use_exp_fn=False):
    '''This method is called by the GUI the first time a Q_Learning
    menu item is selected. It may be called again after the user has
    restarted from the File menu.
    Q_VALUES starts out with all Q-values at 0.0 and a separate key
    for each (s, a) pair.'''
    global STATES, ACTIONS, UQV_callback, Q_VALUES, is_valid_goal_state
    global USE_EXPLORATION_FUNCTION, Terminal_state
    STATES = states
    ACTIONS = actions
    Q_VALUES = q_vals_dict
    UQV_callback = update_q_value_callback
    is_valid_goal_state = goal_test
    Terminal_state = terminal
    USE_EXPLORATION_FUNCTION = use_exp_fn
    if USE_EXPLORATION_FUNCTION:
#*** ADD OR CHANGE CODE HERE ***
         # Change this if you implement an exploration function:
         print("You have not implemented an exploration function")

PREVIOUS_STATE = None
LAST_ACTION = None
def set_starting_state(s):
    '''This is called by the GUI when a new episode starts.
    Do not change this function.'''
    global INITIAL_STATE, PREVIOUS_STATE
    print("In Q_Learn, setting the starting state to "+str(s))
    INITIAL_STATE = s
    PREVIOUS_STATE = s

ALPHA = 0.5
C_ALPHA=0.5
C_EPSILON=0.5
CUSTOM_ALPHA = False
EPSILON = 0.5
CUSTOM_EPSILON = False
GAMMA = 0.9
EPISODES=0
N=0
def set_learning_parameters(alpha, epsilon, gamma):
    ''' Called by the system. Do not change this function.'''
    global ALPHA, EPSILON, CUSTOM_ALPHA, CUSTOM_EPSILON, GAMMA
    ALPHA = alpha
    EPSILON = epsilon
    GAMMA = gamma
    if alpha < 0: CUSTOM_ALPHA = True
    else: CUSTOM_ALPHA = False
    if epsilon < 0: CUSTOM_EPSILON = True
    else: CUSTOM_EPSILON = False

def update_Q_value(previous_state, previous_action, new_value):
    '''Whenever your code changes a value in Q_VALUES, it should
    also call this method, so the changes can be reflected in the
    display.
    Do not change this function.'''
    UQV_callback(previous_state, previous_action, new_value)

def handle_transition(action, new_state, r):
    '''When the user drives the agent, the system will call this function,
    so that you can handle the learning that should take place on this
    transition.'''
    global PREVIOUS_STATE, Q_VALUES, ACTIONS,N, EPISODES
    #print(action)
#*** ADD OR CHANGE CODE HERE ***
    if action=='Exit':
        EPISODES+=1
        print("Episode Number: ",EPISODES)

    Q_VALUES[(PREVIOUS_STATE,action)]+=  ALPHA * (r + GAMMA * max(Q_VALUES[(new_state, a1)] for a1 in ACTIONS) - Q_VALUES[(PREVIOUS_STATE, action)])

    qval=Q_VALUES[(PREVIOUS_STATE,action)]


            # You should call update_Q_value before returning.  E.g.,
    update_Q_value(PREVIOUS_STATE, action, qval)

    print("Transition to state: "+str(new_state)+\
                  "\n with reward "+str(r)+". EPISODE: ",EPISODES)
    PREVIOUS_STATE = new_state
    return # Nothing needs to be returned.

def choose_next_action(s, r, terminated=False):
     '''When the GUI or engine calls this, the agent is now in state s,
     and it receives reward r.
     If terminated==True, it's the end of the episode, and this method
      can just return None after you have handled the transition.

     Use this information to update the q-value for the previous state
     and action pair.

     Then the agent needs to choose its action and return that.

     '''
     global INITIAL_STATE, PREVIOUS_STATE, LAST_ACTION, C_EPSILON, C_ALPHA
     # Unless s is the initial state, compute a new q-value for the
     # previous state and action.
     if not (s==INITIAL_STATE):
         # Compute your update here.
         #new_qval = -99 # A bogus value for now.
#*** ADD OR CHANGE CODE HERE ***

        if CUSTOM_ALPHA==True:
        #N+=1
            Q_VALUES[(PREVIOUS_STATE,LAST_ACTION)]+=  C_ALPHA * 0.99 * (r + GAMMA * max(Q_VALUES[(s, a1)] for a1 in ACTIONS)
              - Q_VALUES[(PREVIOUS_STATE, LAST_ACTION)])

            qval=Q_VALUES[(PREVIOUS_STATE,LAST_ACTION)]

            # You should call update_Q_value before returning.  E.g.,
            update_Q_value(PREVIOUS_STATE, LAST_ACTION, qval)

            print("Transition to state: "+str(s)+\
                  "\n with reward "+str(r)+".")
            #C_ALPHA=C_ALPHA*0.99

        else:

            Q_VALUES[(PREVIOUS_STATE, LAST_ACTION)] += ALPHA * (r + GAMMA * max(Q_VALUES[(s, a1)]
                                           for a1 in ACTIONS) - Q_VALUES[(PREVIOUS_STATE, LAST_ACTION)])

            new_qval = Q_VALUES[(PREVIOUS_STATE,LAST_ACTION)]
            # Save it in the dictionary of Q_VALUES:
            #Q_VALUES[(PREVIOUS_STATE, LAST_ACTION)] = new_qval

            # Then let the Engine and GUI know about the new Q-value.
            update_Q_value(PREVIOUS_STATE, LAST_ACTION, new_qval)

     # Now select an action according to your Q-Learning criteria, such
     # as expected discounted future reward vs exploration.

     if USE_EXPLORATION_FUNCTION:
         # Change this if you implement an exploration function:
#*** ADD OR CHANGE CODE HERE ***
         print("You have not implemented an exploration function")

     # If EPSILON > 0, or CUSTOM_EPSILON is True,
     # then use epsilon-greedy learning here.
     # In order to access q-values, simply get them from the dictionary, e.g.,
     # some_qval = Q_VALUES[(some_state, some_action)]

#*** ADD OR CHANGE CODE HERE ***

     if is_valid_goal_state(s):
        print("It's a goal state.")
        return "Exit"
     elif s==Terminal_state:
        print("It's a Terminal state")
        return None
     else:
        if CUSTOM_EPSILON==True:
            C_EPSILON=C_EPSILON*0.99
            rnum=random.random()
            #v_list=[]
            if rnum>C_EPSILON:
                #print("In loop")
                maxValue=max(Q_VALUES[(s,a)] for a in ACTIONS)
                max_val_list = [key[1] for key,val in Q_VALUES.items() if Q_VALUES[key]==maxValue and key[0]==s]
                if len(max_val_list)>1:

                    some_action=random.choice(max_val_list[:6])
                    LAST_ACTION = some_action
                    PREVIOUS_STATE = s        #    "       "    "   "    "
                    return some_action
                else:
                    some_action=max_val_list[0]
                    LAST_ACTION = some_action # remember this for next time
                    PREVIOUS_STATE = s        #    "       "    "   "    "
                    return some_action

            else:
                action_list=[k for k in ACTIONS if k!='Exit']
                some_action=random.choice(action_list)
                LAST_ACTION = some_action # remember this for next time
                PREVIOUS_STATE = s
                return some_action
        else:
            rnum=random.random()
            #v_list=[]
            if rnum>EPSILON:
                #print("In loop")
                maxValue=max(Q_VALUES[(s,a)] for a in ACTIONS)
                max_val_list = [key[1] for key,val in Q_VALUES.items() if Q_VALUES[key]==maxValue and key[0]==s]
                if len(max_val_list)>1:

                    some_action=random.choice(max_val_list[:6])
                    #print("max list action",some_action)
                    #print("max list",max_val_list[:6])
                    LAST_ACTION = some_action
                    PREVIOUS_STATE = s        #    "       "    "   "    "
                    return some_action
                else:
                    some_action=max_val_list[0]
                    LAST_ACTION = some_action # remember this for next time
                    PREVIOUS_STATE = s        #    "       "    "   "    "
                    return some_action

            else:
                action_list=[k for k in ACTIONS if k!='Exit']
                some_action=random.choice(action_list)
                LAST_ACTION = some_action # remember this for next time
                PREVIOUS_STATE = s
                return some_action



Policy = {}
def extract_policy(S, A):
   '''Return a dictionary mapping states to actions. Obtain the policy
   using the q-values most recently computed.
   Ties between actions having the same (s, a) value can be broken arbitrarily.
   Reminder: goal states should map to the Exit action, and no other states
   should map to the Exit action.
   '''
   global Policy
   Policy = {}
#*** ADD OR CHANGE CODE HERE ***

   for s in S:
      if is_valid_goal_state(s):
        Policy[s]='Exit'
      v_list = [] #empty list to append Q-values
      for a in A:
         if a!='Exit':
            v_list.append(Q_VALUES[(s, a)])
      idx=v_list.index(max(v_list)) #extract index of max element in list
      Policy[s] = A[idx]
   return Policy
