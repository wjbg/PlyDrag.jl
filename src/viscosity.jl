""" viscosity_models.jl

This file defines temperature and rheology models for fluids, including Arrhenius and WLF
temperature-dependent models, and the Cross rheology model with temperature-dependent
parameters.
"""

const GAS_CONSTANT = 8.31446261815324  # J/(mol·K)

abstract type TemperatureModel end
abstract type RheologyModel end


# ================================================================
# Temperature Models
# ================================================================

"""
    Constant{T <: Real}

A model that returns a constant value regardless of the temperature. Useful for parameters
that do not depend on temperature.
"""
struct Constant{T <: Real} <: TemperatureModel
    value::T
end

"""
    Arrhenius{T <: Real}

Temperature-dependent model based on the Arrhenius equation.

The model is defined as:

    A(T) = Aref * exp( (E / R) * (1/T - 1/Tref) ),

with `A` the property of interest at reference temperature `Tref`.

# Fields
- `Aref`: Pre-exponential factor at reference temperature (dimension depends on
          application)
- `Tref`: Reference temperature [K]
- `E`: Activation energy [J/mol]
"""
Base.@kwdef struct Arrhenius{T <: Real} <: TemperatureModel
    Aref::T  # Pre-exponential factor at reference temperature
    Tref::T  # Reference temperature [K]
    E::T     # Activation energy [J/mol]
end

"""
    WLF{T <: Real}

Williams–Landel–Ferry (WLF) temperature dependence model.

The model is defined as:

    val(T) = A * 10^(-C1 * (T - Tr) / (C2 + (T - Tr)))

# Fields
- `A`: Pre-exponential factor (dimension depends on application)
- `Tr`: Reference temperature [K] (often glass transition temperature)
- `C1`: Empirical fitting parameter [-]
- `C2`: Empirical fitting parameter [K]

# Notes
- `T` must be provided in Kelvin.
- Note that the formulation uses base-10 exponentials, consistent with the
  classical WLF equation.
"""
Base.@kwdef struct WLF{T <: Real} <: TemperatureModel
    A::T   # Pre-exponential factor
    Tr::T  # Reference temperature [K] (e.g., glass transition temperature)
    C1::T  # Fitting factor
    C2::T  # Fitting factor
end

"""
    (model::TemperatureModel)(T)

Evaluate the temperature-dependent value of the modeled quantity at temperature `T`.

This function implements the universal interface for all temperature models.
It returns the value of a material property (e.g. viscosity, relaxation time)
as a function of temperature.

# Arguments
- `model`: temperature dependence model (e.g. `Arrhenius`, `WLF`, `Constant`)
- `T`: temperature in Kelvin

# Returns
- Value of the modeled property at temperature `T`

# Notes
All temperature models must implement this interface.
"""
function (model::TemperatureModel)(T::Real)
    return error("Temperature model not implemented for $(typeof(model))")
end

(model::Constant)(T::Real) = model.value

function (model::Arrhenius)(T::Real)
    return model.Aref * exp( (model.E / GAS_CONSTANT) * (1/T - 1/model.Tref) )
end

function (model::WLF)(T::Real)
    return model.A * exp10((-model.C1 * (T - model.Tr)) / (model.C2 + (T - model.Tr)))
end


# ================================================================
# Rheology Models
# ================================================================

"""
    Newtonian{T <: TemperatureModel}

A rheology model where the viscosity depends only on temperature, defined by
a `TemperatureModel`.
"""
struct Newtonian{Tη <: TemperatureModel} <: RheologyModel
    η0::Tη
end

"""
    PowerLaw{TK <: TemperatureModel, Tn <: Real}

Power-law (Ostwald–de Waele) rheology model.

The viscosity is defined as:

    η(γ̇, T) = K(T) * γ̇^(n - 1)

# Fields
- `K`: Consistency index model [Pa·sⁿ]
- `n`: Flow behavior index [-]
"""
Base.@kwdef struct PowerLaw{TK <: TemperatureModel, Tn <: Real} <: RheologyModel
    K::TK
    n::Tn
end

"""
    Cross{Tη0 <: TemperatureModel, Tηinf <: TemperatureModel}

Cross rheology model where viscosity is defined as:

    η(γ̇, T) = ηinf(T) + (η0(T) - ηinf(T)) / (1 + (η0(T) * γ̇/τ)^(1 - n))

# Fields
- `η0`: Temperature-dependent zero-shear viscosity model [Pa⋅s]
- `ηinf`: Temperature-dependent infinite-shear viscosity model [Pa⋅s]
- `τ`: Shear stress at onset of shear-thinning [Pa]
- `n`: Power-law index [-]
"""
Base.@kwdef struct Cross{Tη0 <: TemperatureModel,
                         Tηinf <: TemperatureModel} <: RheologyModel
    η0::Tη0
    ηinf::Tηinf
    τ::Float64
    n::Float64
end

"""
    Carreau{Tη0 <: TemperatureModel, Tηinf <: TemperatureModel, Tλ <: TemperatureModel}

Carreau rheology model where viscosity is defined as:

    η(γ̇, T) = ηinf(T) + (η0(T) - ηinf(T)) * (1 + (λ(T) * γ̇)^2)^((n - 1)/2)

# Fields
- `η0`: Temperature-dependent zero-shear viscosity model [Pa·s]
- `ηinf`: Temperature-dependent infinite-shear viscosity model [Pa·s]
- `λ`: Temperature-dependent time constant model [s]
- `n`: Power-law index [-]
"""
Base.@kwdef struct Carreau{Tη0 <: TemperatureModel,
                           Tηinf <: TemperatureModel,
                           Tλ <: TemperatureModel} <: RheologyModel
    η0::Tη0
    ηinf::Tηinf
    λ::Tλ
    n::Float64
end

"""
    CarreauYasuda{Tη0 <: TemperatureModel, Tηinf <: TemperatureModel, Tλ <: TemperatureModel}

Carreau–Yasuda rheology model where viscosity is defined as:

    η(γ̇, T) = ηinf(T) + (η0(T) - ηinf(T)) * (1 + (λ(T) * γ̇)^a)^((n - 1)/a)

# Fields
- `η0`: Temperature-dependent zero-shear viscosity model [Pa·s]
- `ηinf`: Temperature-dependent infinite-shear viscosity model [Pa·s]
- `λ`: Temperature-dependent time constant model [s]
- `n`: Power-law index [-]
- `a`: Yasuda parameter [-]
"""
struct CarreauYasuda{Tη0 <: TemperatureModel,
                     Tηinf <: TemperatureModel,
                     Tλ <: TemperatureModel} <: RheologyModel
    η0::Tη0
    ηinf::Tηinf
    λ::Tλ
    n::Float64
    a::Float64
end

"""
    (model::RheologyModel)(γ̇, T)

Evaluate the viscosity of a rheological model at shear rate `γ̇` and temperature `T`.

This is the primary constitutive interface for all rheology models. It returns the
viscosity η(γ̇, T) as defined by the specific model implementation.

# Arguments
- `γ̇`: shear rate (must be ≥ 0)
- `T`: temperature in Kelvin

# Returns
- Viscosity η(γ̇, T)

# Notes
All rheology models must implement this interface.
"""
function (model::RheologyModel)(γ̇::Real, T::Real)
    return error("viscosity not implemented for $(typeof(model))")
end

(model::Newtonian)(γ̇::Real, T::Real) = model.η0(T)

function (model::PowerLaw)(γ̇::Real, T::Real)
    K = model.K(T)
    return K * γ̇^(model.n - 1)
end

function (model::Cross)(γ̇::Real, T::Real)
    η0, ηinf = model.η0(T), model.ηinf(T)
    return ηinf + (η0 - ηinf) / (1 + (η0 * γ̇ / model.τ)^(1 - model.n))
end

function (model::Carreau)(γ̇::Real, T::Real)
    η0, ηinf, λ = model.η0(T), model.ηinf(T), model.λ(T)
    return ηinf + (η0 - ηinf) * (1 + (λ * γ̇)^2)^((model.n - 1) / 2)
end

function (model::CarreauYasuda)(γ̇::Real, T::Real)
    η0, ηinf, λ = model.η0(T), model.ηinf(T), model.λ(T)
    return ηinf + (η0 - ηinf) * (1 + (λ * γ̇)^model.a)^((model.n - 1) / model.a)
end


# ================================================================
# Predefined models
# ================================================================

"""
    PEEK

Cross rheology model for the temperature-dependent viscosity of PEEK,
with Arrhenius-type temperature dependence, as measured and fitted by Pierik.

# Usage
- `η = PEEK(γ̇, T)`

# Arguments
- `γ̇`: Shear rate [1/s], must be non-negative
- `T`: Temperature [K]

# Returns
- `η`: Viscosity [Pa⋅s]
"""
PEEK = Cross(
    η0 = Arrhenius(Aref=733.67, Tref=370.0+273.15, E=5.57E4),
    ηinf = Constant(0.0),
    τ = 6.05E4,
    n = 0.38
)

"""
    LMPAEK

Cross rheology model for the temperature-dependent viscosity of LM-PAEK,
with Arrhenius-type temperature dependence, as measured and fitted by Pierik.

# Usage
- `η = LMPAEK(γ̇, T)`

# Arguments
- `γ̇`: Shear rate [1/s], must be non-negative
- `T`: Temperature [K]

# Returns
- `η`: Viscosity [Pa⋅s]
"""
LMPAEK = Cross(
    η0 = Arrhenius(Aref=681.90, Tref=345.0+273.15, E=5.13E4),
    ηinf = Constant(0.0),
    τ = 2.53E5,
    n = 0.42
)

"""
    PPS

Cross rheology model for the temperature-dependent viscosity of PPS,
with Arrhenius-type temperature dependence, as measured and by Grouve.

# Usage
- `η = PPS(γ̇, T)`

# Arguments
- `γ̇`: Shear rate [1/s], must be non-negative
- `T`: Temperature [K]

# Returns
- `η`: Viscosity [Pa⋅s]
"""
PPS = Cross(
    η0 = Arrhenius(Aref=223.21, Tref=300.0+273.15, E=6.86E4),
    ηinf = Constant(0.0),
    τ = 7.41E5,
    n = 0.28
)
