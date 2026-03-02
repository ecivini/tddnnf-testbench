from ddnnife import Ddnnf
import time

nnf_path = "results/tddnnf_compilation_rand_part/data/michelutti_tdds/ldd_randgen/data/ldd_phi_problems_b10_r10_d4_m20_s1234/01/ldd_phi_b10_d4_r10_s1234_01/compilation_output.nnf"

ddnnf = Ddnnf.from_file(nnf_path, None)

assumptions = [1, 10, 9]
projection_vars = []

start = time.time()
print(ddnnf.as_mut().count_multiple(assumptions, variables=projection_vars)[0])
end = time.time()

print("total time:", end - start)
