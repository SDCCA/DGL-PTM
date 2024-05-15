import torch
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from ..util.utils import load_consumption_model
#from dgl_ptm.util.nn_arch import parse_config



def wealth_consumption(model_graph, model_params, device=None, method='pseudo_consumption'):
    # Calculate wealth consumed
    if method == 'pseudo_consumption':
        _pseudo_wealth_consumption(model_graph)
    elif method == 'fitted_consumption':
        _fitted_wealth_consumption(model_graph)
    elif method == 'calc_bellman_consumption':
        _bellman_wealth_consumption(model_graph,model_params)
    elif method == 'estimated_bellman_consumption':
        _nn_bellman_wealth_consumption(model_graph,model_params,device)
    else:
        raise NotImplementedError("Incorrect method received. \
                         Method needs to be 'pseudo_consumption','fitted_consumption', or 'bellman_consumption'")
    

def _pseudo_wealth_consumption(model_graph):
# Placeholder consumption ratio     
    model_graph.ndata['wealth_consumption'] = model_graph.ndata['wealth']*1./3.
    
def _fitted_wealth_consumption(model_graph):
# Estimation based on curve fitted to Bellman equation output
    model_graph.ndata['wealth_consumption'] = 0.64036047*torch.log(model_graph.ndata['wealth'])



# Miscellenious information needed for _bellman_wealth_consumption, a very slow method comparatively speaking
def income_function(k,α,tech): 
    f=α * k**tech['gamma'] - tech['cost']
    return torch.max(f)


class BellmanEquation:
    #Adapted from: https://python.quantecon.org/optgrowth.html
    def __init__(self,
                u,            # utility function
                f,            # production function
                k,            # current state k_t
                θ,            # given shock factor θ
                σ,            # risk averseness
                α,            # human capital
                i_a,          # adaptation investment
                m,            # protection multiplier
                β,            # discount factor
                𝛿,            # depreciation factor 
                tech):       # adaptation table 
                #name="BellmanNarrowExtended"
                

        self.u, self.f, self.k, self.β, self.θ, self.𝛿, self.σ, self.α, self.i_a, self.m, self.tech = u, f, k, β, θ, 𝛿, σ, α, i_a, m, tech

        # Set up grid
        
        startgrid=np.array([1.0e-7,1,2,3,4,5,6,7,8,9,10,k+100])

        ind=np.searchsorted(startgrid, k)
        self.grid=np.concatenate((startgrid[:ind],np.array([k*0.99999, k]),
                                startgrid[ind:]))

        self.grid=self.grid[self.grid>i_a]

        # Identify target state k
        self.index = np.searchsorted(self.grid, k)-1
    def value(self, c, y, v_array):
        """
        Right hand side of the Bellman equation.
        """

        u, f, β, θ, 𝛿, σ, α, i_a, m, tech = self.u, self.f, self.β, self.θ, self.𝛿, self.σ, self.α, self.i_a, self.m, self.tech

        v = interp1d(self.grid, v_array, bounds_error=False, 
                    fill_value="extrapolate")
        return u(c,σ) + β * v((θ + m * (1-θ)) * (f(y,α,tech) - c - i_a + (1 - 𝛿) * y))





def _bellman_wealth_consumption(model_graph, model_params):
    
    def maximize(g, a, b, args):
        """
        From: https://python.quantecon.org/optgrowth.html (similar example 
        https://macroeconomics.github.io/Dynamic%20Programming.html#.ZC13-exBy3I)
        Maximize the function g over the interval [a, b].

        The maximizer of g on any interval is
        also the minimizer of -g.  The tuple args collects any extra
        arguments to g.

        Returns the maximum value and the maximizer.
        """

        objective = lambda x: -g(x, *args)
        result = minimize_scalar(objective, bounds=(a, b), method='bounded')
        maximizer, maximum = result.x, -result.fun
        return maximizer, maximum

    def utility(c, σ, type="isoelastic"):
        if type == "isoelastic":
            if σ ==1:
                return np.log(c)
            else:
                return (c**(1-σ)-1)/(1-σ)

        else:
            print("Unspecified utility function!!!")


    def update_bellman(v, bell):
        """
        From: https://python.quantecon.org/optgrowth.html (similar example
        https://macroeconomics.github.io/Dynamic%20Programming.html#.ZC13-exBy3I)
        
        The Bellman operator.  Updates the guess of the value function
        and also computes a v-greedy policy.

        * bell is an instance of Bellman equation
        * v is an array representing a guess of the value function

        """
        v_new = np.empty_like(v)
        v_greedy = np.empty_like(v)
        
        for i in range(len(bell.grid)):
            y = bell.grid[i]
            # Maximize RHS of Bellman equation at state y
            
            c_star, v_max = maximize(bell.value, min([1e-8,y*0.00001]), 
                                    y-bell.i_a, (y, v))
            #VMG HELP! can anyone check that (1) subtracting i_a and 
            # (2) omitting any grid values less than i_a 
            # will not be problematic? The only thing I can come up with
            # is if i_a is greater than k*0.99999
            # which_bellman() now accounts for that case. Whole thing 
            # could use refinement.
        
            v_new[i] = v_max
            v_greedy[i] = c_star

        return v_greedy, v_new


    def which_bellman(agentinfo):
        """
        Solves bellman for each affordable adaptation option.
        """
        feasible=[]

        for option in torch.transpose(agentinfo['adapt'],0,1):
            if option[1]>=((income_function(agentinfo['k'],agentinfo['α'],tech={'gamma':model_params['tech_gamma'],'cost':model_params['tech_cost']})+(1 - model_params['depreciation'])*agentinfo['k'])*.99998):
                # ensures that the gridpoint
                # just below k, k*0.99999, is included
                pass
            else:
                #  print(f'working theta = {agentinfo.θ + option[0] *\
                #  (1-agentinfo.θ)}, i_a= {option[1]}, k= {agentinfo.k}')
                c,v=solve_bellman(BellmanEquation(u=utility, 
                                f=income_function, k=agentinfo['k'], 
                                θ=agentinfo['θ'], σ=agentinfo['σ'], 
                                α=agentinfo['α'], i_a=option[1].numpy(),m=option[0],
                                β=model_params['discount'], 𝛿=model_params['depreciation'],
                                tech={'gamma':model_params['tech_gamma'],'cost':model_params['tech_cost']}))
                feasible.append([v,c,option[1],option[0]])

        best=min(feasible)

        return best[1],best[2],best[3]

    def solve_bellman(bell,
                    tol=1,
                    min_iter=10,
                    max_iter=1000,
                    verbose=False):
        """
        From: https://python.quantecon.org/optgrowth.html (similar example
        https://macroeconomics.github.io/Dynamic%20Programming.html#.ZC13-exBy3I)
        
        Solve model by iterating with the Bellman operator.

        """

        # Set up loop

        v = bell.u(bell.grid,bell.σ)  # Initial condition
        i = 0
        error = tol + 1

        while (i < max_iter and error > tol) or (i < min_iter):
            v_greedy, v_new = update_bellman(v, bell)
            error = np.abs(v[bell.index] - v_new)[bell.index]
            i += 1
            # if verbose and i % print_skip == 0:
            #     print(f"Error at iteration {i} is {error}.")
            v = v_new

        if error > tol:
            print(f"{bell.name} failed to converge for k={bell.k}, α = {bell.α},σ ={bell.σ}, i_a={bell.i_a}, and modified θ = {bell.θ + bell.m * (1-bell.θ)}!")
        elif verbose:
            print(f"Converged in {i} iterations.")
            print(f"Effective k and new c {np.around(bell.grid[bell.index],3),v_greedy[bell.index]}")
            

        return v_greedy[bell.index],v[bell.index]
    

    for i in range(model_graph.num_nodes()):
        agentinfo = {'u':utility, 'f':income_function, 'β':model_params['discount'], 'θ':model_graph.ndata['theta'][i], '𝛿':model_params['depreciation'], 'σ':model_graph.ndata['sigma'][i].numpy(), 'α': model_graph.ndata['alpha'][i],'k':model_graph.ndata['wealth'][i],'adapt': model_graph.ndata['a_table'][i]}
        model_graph.ndata['wealth_consumption'][i], model_graph.ndata['i_a'][i], model_graph.ndata['m'][i] = which_bellman(agentinfo)



def  _nn_bellman_wealth_consumption(model_graph,model_params, device):
    ''' Estimation of consumption and i_a using a pytorch neural network trained on Bellman equation output. 
    Currently only works with a single model for all agents under the four input two output configuration. 
    The entire surrogate model must be saved at nn_path and the architecture of the model specified in nn_arch.py.''' 

    if model_params['nn_path']==None:
        print("No consumption model path provided!")
    
    #load model  
    
    estimator,scale = load_consumption_model(model_params['nn_path'],device)  

    estimator.eval()

    input = torch.cat((model_graph.ndata['alpha'].unsqueeze(1), model_graph.ndata['wealth'].unsqueeze(1), model_graph.ndata['sigma'].unsqueeze(1), model_graph.ndata['theta'].unsqueeze(1)), dim=1) 
    
    #forward pass to get predictions
    with torch.no_grad():

        pred=estimator(input)

    print(" went forward, writing values")

    
    model_graph.ndata['m'],model_graph.ndata['i_a']=model_graph.ndata['a_table'][torch.arange(model_graph.ndata['a_table'].size(0)),:,torch.argmin(torch.abs(pred[:, 0].unsqueeze(1) - model_graph.ndata['a_table'][:,1,:]), dim=1)].unbind(dim=1)
    
    print("Cleaning output and checking for violations")

    #Clean Consumption
    model_graph.ndata['wealth_consumption']=(pred[:,1]*scale).clamp_(min=0)

    # Check for violations
    # A violation occurs when depreciated k + income - consumption - i_a is less than or equal to 0
    violation = (1-model_params['depreciation'])*model_graph.ndata['wealth']+model_graph.ndata['income']-model_graph.ndata['wealth_consumption']-model_graph.ndata['i_a']<=0


    if torch.sum(violation)!=0:
        # Violation type 1: i_a exceeds depreciated k + income
        violation_i_a = (1-model_params['depreciation'])*model_graph.ndata['wealth']+model_graph.ndata['income']-model_graph.ndata['i_a']<=0
        # Violation type 2: consumption exceeds k
        #violation_consumption = model_graph.ndata['wealth']-model_graph.ndata['wealth_consumption']<=0

        print(f"Agents in violation: {torch.sum(violation)}")
        print(f"...because of i_a: {torch.sum(violation_i_a)}")

        # setting i_a to 0 for type 1 violations
        model_graph.ndata['m'][torch.nonzero(violation_i_a, as_tuple=False)]=0
        model_graph.ndata['i_a'][torch.nonzero(violation_i_a, as_tuple=False)]=0

        # redetermine violations with updated i_a
        violation = (1-model_params['depreciation'])*model_graph.ndata['wealth']+model_graph.ndata['income']-model_graph.ndata['wealth_consumption']-model_graph.ndata['i_a']<=0
        violation_i_a = (1-model_params['depreciation'])*model_graph.ndata['wealth']+model_graph.ndata['income']-model_graph.ndata['i_a']<=0

        model_graph.ndata['wealth_consumption'][torch.nonzero(violation, as_tuple=False)]=((1-model_params['depreciation'])*model_graph.ndata['wealth'][torch.nonzero(violation, as_tuple=False)]+model_graph.ndata['income'][torch.nonzero(violation, as_tuple=False)]-model_graph.ndata['i_a'][torch.nonzero(violation, as_tuple=False)])*0.99


    violation = (1-model_params['depreciation'])*model_graph.ndata['wealth']+model_graph.ndata['income']-model_graph.ndata['wealth_consumption']-model_graph.ndata['i_a']<=0

    if torch.sum(violation)!=0:

        print(f"Something has gone terribly wrong! Still {torch.sum(violation)} violations.")












    









