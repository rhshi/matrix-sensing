import argparse
from symmetric import SymmetricEmpirical, SymmetricPopulation
from asymmetric import AsymmetricEmpirical, AsymmetricPopulation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_file', type=str)
    parser.add_argument('--mode', type=str, default="sym", choices=["sym", "asym"])
    parser.add_argument('--risk', type=str, default="empirical", choices=["empirical", "population"])
    parser.add_argument('-d', type=int, default=100)
    parser.add_argument('-r', type=int, default=5)
    parser.add_argument('--eta', type=float, default=0.0025)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--iters', type=int, default=int(1e4))
    parser.add_argument('--log_freq', type=int, default=250)

    args = parser.parse_args()

    if args.mode == "sym":
        if args.risk == "empirical":
            sim = SymmetricEmpirical(args.d, args.r, args.eta, args.log_file)
        elif args.risk == "population":
            sim = SymmetricPopulation(args.d, args.r, args.eta, args.log_file)
    elif args.mode == "asym":
        if args.risk == "empirical":
            sim = AsymmetricEmpirical(args.d, args.r, args.eta, args.log_file)
        elif args.risk == "population":
            sim = AsymmetricPopulation(args.d, args.r, args.eta, args.log_file)
    
    U = sim.go(args.alpha, args.iters, args.log_freq)

    return


if __name__ == "__main__":
    main()