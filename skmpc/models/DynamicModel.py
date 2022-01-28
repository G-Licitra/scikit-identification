#%%
import pandas as pd
import casadi as ca


class System:
    """Casadi Class System
    Formulation:
    \dot(x(t)) = f(x(x), u(t), \theta)
    y(t) = g(f(x))

    with: 
    - $x(t) \in \Reˆ{n_{x}}$ differential states
    - $u(t) \in \Reˆ{n_{u}}$ control inputs
    - $\theta \in \Reˆ{n_{\theta}}$ model parameters
    - $\dot(x(t)) \in \Reˆ{n_{x}}$ model dynamics defined as Ordinary Differential Equation (ODE)
    - $y(t) \in \Reˆ{n_{y}}$$ model output
    
    Parameters
    ---------
    input1: array-like
        explain input1
    input2: None or str (default: None)
        explain input2
    figsize: tuple (default: (11.7, 8.27))
        explain figsize

    Returns
    ---------
    p_plot: Figure

    Examples
    ---------

    >>> input1 = np.random.rand(100)
    >>> input2 = 2
    >>> p_plot = func(input1, input2)

    """

    # class variable: every instance will inherit this value
    #nationality = "italy" 

    def __init__(self, 
                 states, 
                 inputs, 
                 parameters=None, 
                 model_dynamics=None, 
                 output=None, 
                 name_state=None, 
                 name_input=None, 
                 name_output=None):
        
        """Constructor. It runs every instance is created"""
        self.states = states
        self.inputs = inputs
        self.parameters = parameters
     
        # get dimentions 
        self.nx = states.shape[0] # differential states
        self.nu = inputs.shape[0] # control input
        self.np = -1 if parameters == None else parameters.shape[0] # model parameters
           
        self.model_dynamics = model_dynamics
        
        # if output y(t) is not specified, then set y = x(t) 
        self.output = states if output == None else output
        self.ny = self.nx if output == None else output.shape[0] # model parameters  
    
        # Construct Model
        # Form an ode function
        print(self.np) 
        if self.np == -1:    
            self.model = ca.Function('model',
                                    [self.states, self.inputs],
                                    [self.model_dynamics, self.output],
                                    ['x(y)','u(t)'], ['xdot(t) = f(x(t),u(t))', 'y(t) = g(x(t))'] )
        else:
            self.model = ca.Function('model',
                                    [self.states, self.inputs, self.parameters],
                                    [self.model_dynamics, self.output],
                                    ['x(y)','u(t)','theta'], ['xdot(t) = f(x(t),u(t),theta)', 'y(t) = g(x(t))'] )
        
    
    def print_dimensions(self):
        self.model.print_dimensions()
        
    def print_ode(self):
        print(self.model_dynamics)
    
    def print_output(self):
        print(self.output)    

    def evaluate(self, x0, u0, theta):
        (rhs_num, y_num) = self.model(x0, u0, theta)    
        return (rhs_num, y_num)
          
    def __repr__(self):
        """"
        If you don't overload __str__ when running:
        >>> print(sue) # will show something like
        <__main__.Person object at 0x7fa3288ed9d0> 
        __repr__ and __str__ are run automatically everytime an instance is converted to its print string.
        o __str__ are used for more user friedly info
        o __repr__ is used to provide extra details to developers
        """
        return "self.model.print_dimensions()"  # string to print


#%%
#if __name__ == "__main__":  # when run for testing only

x = ca.MX.sym("x", 2)
u = ca.MX.sym("u", 2)
param = ca.MX.sym("theta", 2)
    
x1, x2 = x[0], x[1]
u1, u2 = u[0], u[1]
ka, kb = param[0], param[1]

# xdot = f(x,u,p) <==> rhs = f(x,u,p)

rhs = ca.vertcat(u1-ka*x1, 
                u1*u2/x1-u1*x2/x1-kb*x2)

sys = System(states=x, inputs=u, parameters=param, model_dynamics=rhs)

# show special class attribute
#print(sys.__class__)                # Show bob's class and his name
#print(sys.__class__.__bases__)

# show attribute (the one defined in __init__)
#for key in sys.__dict__:
#    print(key, "=>", sys.__dict__[key]) # 1st way
#    print(key, "=>", getattr(sys, key))  # 2nd way useful to catch exception
# %%

sys.print_dimensions()

# %%
sys.print_ode()
# %%
# numerical evaluation ===================================================
x_test     = [0.1,-0.1]
u_test     = [0.2,-0.1]
theta_test = [0.1,0.5]
rhs_num, y_num  = sys.evaluate(x_test,u_test,theta_test)

print(f"rhs = {rhs_num}, \ny = {y_num}")
