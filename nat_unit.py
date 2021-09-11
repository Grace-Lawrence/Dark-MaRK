import astropy.units as units
import astropy.constants as const

def to_natural(quant, verbose=False):
    Name_unitlist = ["mass", "length", "time", "energy", "momentum", "velocity", "angular momentum", "cross-section", "force"]
    SI_unitlist = [units.kg, units.m, units.s, units.kg * units.m**2 / units.s**2, units.kg * units.m / units.s, units.m / units.s, units.kg * units.m**2 / units.s, units.m**2, units.kg * units.m / units.s**2]
    Nat_unitlist = [units.GeV, units.GeV**-1, units.GeV**-1, units.GeV, units.GeV, units.dimensionless_unscaled, units.dimensionless_unscaled, units.GeV**-2, units.GeV**2]
    Nat_convlist = [const.c**2, (const.c*const.hbar)**-1, (const.hbar)**-1, 1., const.c, (const.c)**-1, (const.hbar)**-1, (const.c*const.hbar)**-2, (const.c*const.hbar)**2]
    
    for SI, Nat, Name, conv in zip(SI_unitlist,Nat_unitlist,Name_unitlist,Nat_convlist):
        testunit = quant
        try:
            testunit.to(SI)
        except:
            testunit = -1
            pass 
        else:
            if verbose:
                print("It is", Name)
                print("Convert ", SI, " to ",Nat)            
            testunit = (testunit * conv)
            testunit = testunit.cgs
            testunit = testunit.to(Nat)
            break
    return testunit # If I never find the variable conversion

