""" viscosity_models.jl

This file defines temperature and rheology models for fluids,
including Arrhenius and WLF temperature shift factors, and the
Cross rheology model with temperature-dependent parameters.
"""

const GAS_CONSTANT = 8.31446261815324  # J/(mol·K)

abstract type TemperatureModel end
abstract type RheologyModel end

"""
    Arrhenius{T <: Real}

Temperature-dependent model based on the Arrhenius equation.

The shift factor is defined as:

    A * exp(E / (R * T))

# Fields
- `A`: Pre-exponential factor (dimension depends on application)
- `E`: Activation energy [J/mol]

# Notes
- `T` must be provided in Kelvin.
- Note that the exponent is positive.
- Commonly used to describe thermally activated processes such as
  viscosity, reaction rates, or diffusion.
"""
Base.@kwdef struct Arrhenius{T <: Real} <: TemperatureModel
    A::T  # Pre-exponential factor
    E::T  # Activation energy [J/mol]
end

"""
    WLF{T <: Real}

Williams–Landel–Ferry (WLF) temperature dependence model.

The shift factor is defined as:

    A * 10^(-C1 * (T - Tr) / (C2 + (T - Tr)))

# Fields
- `A`: Pre-exponential factor (dimension depends on application)
- `Tr`: Reference temperature [K] (often glass transition temperature)
- `C1`: Empirical fitting parameter [-]
- `C2`: Empirical fitting parameter [K]

# Notes
- `T` must be provided in Kelvin.
- The formulation uses base-10 exponentials, consistent with the
  classical WLF equation.
- The model is typically valid near the reference temperature.
"""
Base.@kwdef struct WLF{T <: Real} <: TemperatureModel
    A::T   # Pre-exponential factor
    Tr::T  # Reference temperature [K] (e.g., glass transition temperature)
    C1::T  # Fitting factor
    C2::T  # Fitting factor
end

"""
    Constant{T <: Real}

A model that returns a constant value regardless of the temperature.
Useful for parameters that do not depend on temperature.
"""
Base.@kwdef struct Constant{T <: Real} <: TemperatureModel
    value::T
end

"""
    shift_factor(model::TemperatureModel, T::Real)

Evaluate the temperature-dependent shift factor at temperature `T`.

# Arguments
- `model`: Temperature model (e.g. `Arrhenius`, `WLF`, `Constant`)
- `T`: Temperature [K]

# Returns
- Multiplicative factor used to scale material response (e.g. viscosity
  or relaxation time)

# Notes
- `T` must be given in Kelvin.
- The returned value is typically used to scale viscosity, relaxation time,
  or reaction rates.
"""
shift_factor(model::TemperatureModel, T::Real) =
    return error("shift_factor not implemented for $(typeof(model))")

shift_factor(model::Constant, T::Real) = return model.value

function shift_factor(model::Arrhenius, T::Real)
    return model.A * exp(model.E / (GAS_CONSTANT * T))
end

function shift_factor(model::WLF, T::Real)
    return model.A * 10^((-model.C1 * (T - model.Tr)) / (model.C2 + (T - model.Tr)))
end

"""
    CrossModel{T <: Real, T0 <: TemperatureModel, Tinf <: TemperatureModel, Tλ <: TemperatureModel}

Cross rheology model with temperature-dependent parameters.

The viscosity is defined as:

    η(γ̇, T) = ηinf(T) + (η0(T) - ηinf(T)) / (1 + (λ(T) * γ̇)^(1 - n))

# Fields
- `η0`: Zero-shear viscosity model
- `ηinf`: Infinite-shear viscosity model
- `λ`: Relaxation time model
- `n`: Power-law index [-]
"""
Base.@kwdef struct CrossModel{
    T <: Real,
    T0 <: TemperatureModel,
    Tinf <: TemperatureModel,
    Tλ <: TemperatureModel,
} <: RheologyModel
    η0::T0
    ηinf::Tinf
    λ::Tλ
    n::T
end

"""
    ConstantViscosity{T <: Real}

A rheology model that returns a constant viscosity regardless of shear rate
or temperature.
"""
Base.@kwdef struct ConstantViscosity{T <: Real} <: RheologyModel
    value::T
end

viscosity(model::ConstantViscosity, γ̇::Real, T::Real) = model.value

"""
    Newtonian{T <: TemperatureModel}

A rheology model where the viscosity depends only on temperature, defined by
a `TemperatureModel`.
"""
Base.@kwdef struct Newtonian{T <: TemperatureModel} <: RheologyModel
    η::T
end

viscosity(model::Newtonian, γ̇::Real, T::Real) = shift_factor(model.η, T)

"""
    viscosity(model::RheologyModel, γ̇::Real, T::Real)

Evaluate the temperature and shear-rate-dependent viscosity.
"""
viscosity(model::RheologyModel, γ̇::Real, T::Real) =
    return error("viscosity not implemented for $(typeof(model))")

function viscosity(model::CrossModel, γ̇::Real, T::Real)
    η0 = shift_factor(model.η0, T)
    ηinf = shift_factor(model.ηinf, T)
    λ = shift_factor(model.λ, T)
    return ηinf + (η0 - ηinf) / (1 + (λ * γ̇)^(1 - model.n))
end

# -------------------------------------------------
# Predefined Models
# -------------------------------------------------

"""
    PPS

Cross model parameters for Polyphenylene Sulfide (PPS).
"""
const PPS = CrossModel(
    η0 = Arrhenius(A = 1.25e-4, E = 6.86e4),
    ηinf = Constant(value = 0.0),
    λ = Arrhenius(A = 2.21e-8, E = 4.50e4),
    n = 0.28,
)
