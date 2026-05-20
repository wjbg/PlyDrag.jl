""" viscosity_models.jl

This file defines temperature and rheology models for fluids, including Arrhenius and WLF
temperature shift factors, and the Cross rheology model with temperature-dependent
parameters. """

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

The shift factor is defined as:

    A * exp(E / (R * T))

# Fields
- `A`: Pre-exponential factor (dimension depends on application)
- `E`: Activation energy [J/mol]

# Notes
- `T` must be provided in Kelvin.
- Note that the exponent is positive.
- Commonly used to describe thermally activated processes such as viscosity,
  reaction rates, or diffusion.
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

Evaluate the temperature-dependent scaling factor `a_T` at temperature `T`.

This function implements the universal interface for all temperature models.
It returns a multiplicative factor used to scale material properties such as
viscosity, relaxation time, or other thermally activated quantities.

# Arguments
- `model`: temperature dependence model (e.g. `Arrhenius`, `WLF`, `Constant`)
- `T`: temperature in Kelvin

# Returns
- Dimensionless or model-specific scaling factor `a_T(T)`

# Notes
All temperature models must implement this interface.
"""
function (model::TemperatureModel)(T::Real)
    return error("Temperature shift not implemented for $(typeof(model))")
end

(model::Constant)(T::Real) = model.value

function (model::Arrhenius)(T::Real)
    return model.A * exp(model.E / (GAS_CONSTANT * T))
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
    η::Tη
end

"""
    CrossModel{T <: Real,
               T0 <: TemperatureModel,
               Tinf <: TemperatureModel,
               Tλ <: TemperatureModel}

Cross rheology model with temperature-dependent parameters.

The viscosity is defined as:

    η(γ̇, T) = ηinf(T) + (η0(T) - ηinf(T)) / (1 + (λ(T) * γ̇)^(1 - n))

# Fields
- `η0`: Zero-shear viscosity model
- `ηinf`: Infinite-shear viscosity model
- `λ`: Relaxation time model
- `n`: Power-law index [-]
"""
Base.@kwdef struct CrossModel <: RheologyModel
    η0::TemperatureModel
    ηinf::TemperatureModel
    λ::TemperatureModel
    n::Float64
end

"""
    (model::RheologyModel)(γ̇, T)

Evaluate the viscosity of a rheological model at shear rate `γ̇` and temperature `T`.

This is the primary constitutive interface for all rheology models. It returns the
viscosity η(γ̇, T) as defined by the specific model implementation.

# Arguments
- `γ̇`: shear rate
- `T`: temperature in Kelvin

# Returns
- Viscosity η(γ̇, T)

# Notes
All rheology models must implement this interface.
"""
function (model::RheologyModel)(γ̇::Real, T::Real)
    return error("viscosity not implemented for $(typeof(model))")
end

(model::Newtonian)(γ̇::Real, T::Real) = model.η(T)

function (model::CrossModel)(γ̇::Real, T::Real)
    η0 = model.η0(T)
    ηinf = model.ηinf(T)
    λ = model.λ(T)
    x = max(λ * γ̇, 0)
    return ηinf + (η0 - ηinf) / (1 + x^(1 - model.n))
end

"""
    η(model::RheologyModel, γ̇::Real, T::Real)

Convenience function for the viscosity.
"""
η(model::RheologyModel, γ̇::Real, T::Real) = model(γ̇, T)


# ================================================================
# Predefined models
# ================================================================

"""
    PPS

Cross model parameters for Polyphenylene Sulfide (PPS).
"""
const PPS = CrossModel(
    η0 = Arrhenius(A = 1.25e-4, E = 6.86e4),
    ηinf = Constant(0.0),
    λ = Arrhenius(A = 2.21e-8, E = 4.50e4),
    n = 0.28,
)
