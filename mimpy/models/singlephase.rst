
Single-Phase Derivation
----------------------

Derivation of the single-phase solution using the Mimetic Finite Difference Method. 
We start with the equation for single phase slightly compressible fluid:

.. math:: 
     \phi \frac{\partial \rho}{\partial t} = \nabla \cdot \frac{\rho}{\mu}K \nabla p + f


Assuming slight compressibilty of the fluid, we have:

.. math::
    \rho(p) \approx \rho^0(1+c_f(p-p^0))

Resulting in, 

.. math::
     \phi \rho^0 c_f \frac{\partial p}{\partial t} = \nabla \cdot \frac{\rho}{\mu}K \nabla p + f

We rewrite this equation in flux form as, 

.. math::
     \begin{align}
     \phi \rho^0 c_f \frac{\partial p}{\partial t} &= -\nabla \cdot v + f\\
     v &= -\frac{\rho}{\mu}K \nabla p 
     \end{align}

Recall the Mimetic method constructs operators :math:`\mathcal{G} \text{ and } \mathcal{DIV}` that approximate 
the continuous forms:

.. math::
    \begin{align}
    \mathcal{DIV} &\approx \nabla \cdot\\
    \mathcal{G} &\approx \frac{\rho}{\mu}K \nabla
    \end{align}

that satify the adjoint relationship:

.. math::
    [p_h, \mathcal{DIV} v_h]_{Q} = -[\mathcal{G} p_h, v_h]_{X} \,\, \forall v_h \in X \text{ and } \forall p_h \in Q

Substituting :math:`\mathcal{G} \text{ and } \mathcal{DIV}` into our equation, we get:

.. math::
     \begin{align}
     \phi \rho^0 c_f \frac{\partial p_h}{\partial t} &=  -\mathcal{DIV }v_h + f^I\\
     v_h &= -\mathcal{G} p_h 
     \end{align}
   
At this point, we have a spatial discretization of the problem.
However, recall that computing operator :math:`\mathcal{G}` explicitly can be difficult as it may 
be a dense matrix. To avoid this problem, we use the adjoint relationship to rewrite the problem 
without explicity computing :math:`\mathcal{G}`:

.. math::
    \begin{align}
     v_h &= -\mathcal{G} p_h \\
     [v_h, u_h]_X &= -[\mathcal{G} p_h, u_h]_X \,\,\, \forall u_h \in X \\
     [v_h, u_h]_X &= [p_h, \mathcal{DIV} u_h]_{Q} \,\,\, \forall u_h \in X
    \end{align}

We call this the "weak" representation of the problem. We can directly write 
the conservation law equation in this form, giving us:

.. math::
    \begin{align}
    \phi \rho^0 c_f \frac{\partial p_h}{\partial t} &=  -\mathcal{DIV }v_h + f^I\\
    [\phi \rho^0 c_f \frac{\partial p_h}{\partial t}, q_h]_Q &=  -[\mathcal{DIV }v_h + f^I, q_h]_Q \forall q_h \in Q
    \end{align}


All togther, we write the problem out in weak form: Find :math:`v_h \in X` and :math:`p_h \in Q` such that: 

.. math::
   \begin{align}
   [v_h, u_h]_X &= [p_h, \mathcal{DIV} u_h]_{Q} \,\,\, \forall u_h \in X\\
   [\phi \rho^0 c_f \frac{\partial p_h}{\partial t}, q_h]_Q &=  -[\mathcal{DIV }v_h + f^I, q_h]_Q \,\,\,\forall q_h \in Q
   \end{align}
 

We are now left with an ODE, so our next step is to solve for the derivative in time. There are plenty of options here, 
one of the simplest thing we can do is to use the Backward-Euler method:

.. math:: 
   \begin{align}
   [v^{n+1}_h, u_h]_X &= [p^{n+1}_h, \mathcal{DIV} u_h]_{Q} \,\,\, \forall u_h \in X\\
   [\phi \rho^0 c_f \frac{p^{n+1}_h-p_h^n}{ \Delta t}, q_h]_Q &=  -[\mathcal{DIV }v^{n+1}_h + f^I, q_h]_Q \,\,\,\forall q_h \in Q
   \end{align}


If we group all the unknowns on the left, we are left with:


Find :math:`v_h \in X` and :math:`p_h \in Q` such that:

.. math:: 
   \begin{align}
   [v^{n+1}_h, u_h]_X - [p^{n+1}_h, \mathcal{DIV} u_h]_{Q}&= 0 \,\,\, \forall u_h \in X\\
   [\mathcal{DIV }v^{n+1}_h, q_h]_Q + [\frac{\phi \rho^0 c_f}{ \Delta t}p^{n+1}_h, q_h]_Q &= 
   [\frac{\phi \rho^0 c_f}{\Delta t}p_h^n + f^I, q_h]_Q \,\,\,\forall q_h \in Q
   \end{align}

This problem can be rewritten as a linear system of equations:

.. math::
   \left(\begin{array}{cc}
   M& D^*\\
   D& C 
   \end{array}\right)
   \left(\begin{array}{c}
   v^{n+1}_h\\
   p^{n+1}_h
   \end{array}\right) = 
   \left(\begin{array}{c}
   0 \\
   f
   \end{array}\right)  
   
.. note::
    Notice that matrix :math:`M` is dependent on :math:`\frac{\rho}{\mu}K`. Since the density 
    term :math:`\rho` is a function of pressure, we technically have a nonlinear system of equations. 
    However, since the compressibility is so small, its effects on the velocity term can be 
    negligable. For this reason, the calculation of density used for building :math:`M` is lagged 
    from the previous time step: :math:`M(\frac{\rho(p^n)}{\mu}K)`. 


Nomenclature:

.. math::
    \begin{align}
    p &= \text{pressure [Pa]}\\
    t &= \text{time [s]}\\
    \phi &= \text{porosity [dimensionless]}\\
    \rho &= \text{density [kg/m$^3$]}\\
    \mu &= \text{viscosity [Pa$\cdot$s]}\\
    f &= \text{source [kg/s]}\\
    p^0 &= \text{reference pressure [Pa]}\\
    \rho^0 &= \text{reference density [kg/m$^3$]} \\
    c_f &= \text{compressibility [Pa$^-1$]}
    \end{align}









    
