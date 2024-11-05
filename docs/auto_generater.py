# -*- coding: utf-8 -*-

import importlib
import inspect
import os

block_list = ['test', 'register_pytree_node', 'call', 'namedtuple', 'jit', 'wraps', 'index', 'function']


def get_class_funcs(module):
    classes, functions, others = [], [], []
    # Solution from: https://stackoverflow.com/questions/43059267/how-to-do-from-module-import-using-importlib
    if "__all__" in module.__dict__:
        names = module.__dict__["__all__"]
    else:
        names = [x for x in module.__dict__ if not x.startswith("_")]
    for k in names:
        data = getattr(module, k)
        if not inspect.ismodule(data) and not k.startswith("_"):
            if inspect.isfunction(data):
                functions.append(k)
            elif isinstance(data, type):
                classes.append(k)
            else:
                others.append(k)

    return classes, functions, others


def _write_module(module_name, filename, header=None, template=False):
    module = importlib.import_module(module_name)
    classes, functions, others = get_class_funcs(module)

    fout = open(filename, 'w')
    # write header
    if header is None:
        header = f'``{module_name}`` module'
    fout.write(header + '\n')
    fout.write('=' * len(header) + '\n\n')
    fout.write(f'.. currentmodule:: {module_name} \n')
    fout.write(f'.. automodule:: {module_name} \n\n')

    # write autosummary
    fout.write('.. autosummary::\n')
    if template:
        fout.write('   :template: classtemplate.rst\n')
    fout.write('   :toctree: generated/\n\n')
    for m in functions:
        fout.write(f'   {m}\n')
    for m in classes:
        fout.write(f'   {m}\n')
    for m in others:
        fout.write(f'   {m}\n')

    fout.close()


def _write_submodules(module_name, filename, header=None, submodule_names=(), section_names=()):
    fout = open(filename, 'w')
    # write header
    if header is None:
        header = f'``{module_name}`` module'
    else:
        header = header
    fout.write(header + '\n')
    fout.write('=' * len(header) + '\n\n')
    fout.write(f'.. currentmodule:: {module_name} \n')
    fout.write(f'.. automodule:: {module_name} \n\n')

    # whole module
    for i, name in enumerate(submodule_names):
        module = importlib.import_module(module_name + '.' + name)
        classes, functions, others = get_class_funcs(module)

        fout.write(section_names[i] + '\n')
        fout.write('-' * len(section_names[i]) + '\n\n')

        # write autosummary
        fout.write('.. autosummary::\n')
        fout.write('   :toctree: generated/\n')
        fout.write('   :nosignatures:\n')
        fout.write('   :template: classtemplate.rst\n\n')
        for m in functions:
            fout.write(f'   {m}\n')
        for m in classes:
            fout.write(f'   {m}\n')
        for m in others:
            fout.write(f'   {m}\n')

        fout.write(f'\n\n')

    fout.close()


def _write_subsections(module_name,
                       filename,
                       subsections: dict,
                       header: str = None):
    fout = open(filename, 'w')
    header = f'``{module_name}`` module' if header is None else header
    fout.write(header + '\n')
    fout.write('=' * len(header) + '\n\n')
    fout.write(f'.. currentmodule:: {module_name} \n')
    fout.write(f'.. automodule:: {module_name} \n\n')

    fout.write('.. contents::' + '\n')
    fout.write('   :local:' + '\n')
    fout.write('   :depth: 1' + '\n\n')

    for name, values in subsections.items():
        fout.write(name + '\n')
        fout.write('-' * len(name) + '\n\n')
        fout.write('.. autosummary::\n')
        fout.write('   :toctree: generated/\n')
        fout.write('   :nosignatures:\n')
        fout.write('   :template: classtemplate.rst\n\n')
        for m in values:
            fout.write(f'   {m}\n')
        fout.write(f'\n\n')

    fout.close()


def _write_subsections_v2(module_path,
                          out_path,
                          filename,
                          subsections: dict,
                          header: str = None):
    fout = open(filename, 'w')
    header = f'``{out_path}`` module' if header is None else header
    fout.write(header + '\n')
    fout.write('=' * len(header) + '\n\n')
    fout.write(f'.. currentmodule:: {out_path} \n')
    fout.write(f'.. automodule:: {out_path} \n\n')

    fout.write('.. contents::' + '\n')
    fout.write('   :local:' + '\n')
    fout.write('   :depth: 1' + '\n\n')

    for name, subheader in subsections.items():
        module = importlib.import_module(f'{module_path}.{name}')
        classes, functions, others = get_class_funcs(module)

        fout.write(subheader + '\n')
        fout.write('-' * len(subheader) + '\n\n')
        fout.write('.. autosummary::\n')
        fout.write('   :toctree: generated/\n')
        fout.write('   :nosignatures:\n')
        fout.write('   :template: classtemplate.rst\n\n')
        for m in functions:
            fout.write(f'   {m}\n')
        for m in classes:
            fout.write(f'   {m}\n')
        for m in others:
            fout.write(f'   {m}\n')
        fout.write(f'\n\n')

    fout.close()


def _write_subsections_v3(module_path,
                          out_path,
                          filename,
                          subsections: dict,
                          header: str = None):
    fout = open(filename, 'w')
    header = f'``{out_path}`` module' if header is None else header
    fout.write(header + '\n')
    fout.write('=' * len(header) + '\n\n')
    fout.write(f'.. currentmodule:: {out_path} \n')
    fout.write(f'.. automodule:: {out_path} \n\n')

    fout.write('.. contents::' + '\n')
    fout.write('   :local:' + '\n')
    fout.write('   :depth: 2' + '\n\n')

    for section in subsections:
        fout.write(subsections[section]['header'] + '\n')
        fout.write('-' * len(subsections[section]['header']) + '\n\n')

        fout.write(f'.. currentmodule:: {out_path}.{section} \n')
        fout.write(f'.. automodule:: {out_path}.{section} \n\n')

        for name, subheader in subsections[section]['content'].items():
            module = importlib.import_module(f'{module_path}.{section}.{name}')
            classes, functions, others = get_class_funcs(module)

            fout.write(subheader + '\n')
            fout.write('~' * len(subheader) + '\n\n')
            fout.write('.. autosummary::\n')
            fout.write('   :toctree: generated/\n')
            fout.write('   :nosignatures:\n')
            fout.write('   :template: classtemplate.rst\n\n')
            for m in functions:
                fout.write(f'   {m}\n')
            for m in classes:
                fout.write(f'   {m}\n')
            for m in others:
                fout.write(f'   {m}\n')
            fout.write(f'\n\n')

    fout.close()


def _write_subsections_v4(module_path,
                          filename,
                          subsections: dict,
                          header: str = None):
    fout = open(filename, 'w')
    header = f'``{module_path}`` module' if header is None else header
    fout.write(header + '\n')
    fout.write('=' * len(header) + '\n\n')

    fout.write('.. contents::' + '\n')
    fout.write('   :local:' + '\n')
    fout.write('   :depth: 1' + '\n\n')

    for name, (subheader, out_path) in subsections.items():

        module = importlib.import_module(f'{module_path}.{name}')
        classes, functions, others = get_class_funcs(module)

        fout.write(subheader + '\n')
        fout.write('-' * len(subheader) + '\n\n')

        fout.write(f'.. currentmodule:: {out_path} \n')
        fout.write(f'.. automodule:: {out_path} \n\n')

        fout.write('.. autosummary::\n')
        fout.write('   :toctree: generated/\n')
        fout.write('   :nosignatures:\n')
        fout.write('   :template: classtemplate.rst\n\n')
        for m in functions:
            fout.write(f'   {m}\n')
        for m in classes:
            fout.write(f'   {m}\n')
        for m in others:
            fout.write(f'   {m}\n')
        fout.write(f'\n\n')

    fout.close()


def _get_functions(obj):
    return set([n for n in dir(obj)
                if (n not in block_list  # not in blacklist
                    and callable(getattr(obj, n))  # callable
                    and not isinstance(getattr(obj, n), type)  # not class
                    and n[0].islower()  # starts with lower char
                    and not n.startswith('__')  # not special methods
                    )
                ])


def _import(mod, klass=None, is_jax=False):
    obj = importlib.import_module(mod)
    if klass:
        obj = getattr(obj, klass)
        return obj, ':meth:`{}.{}.{{}}`'.format(mod, klass)
    else:
        if not is_jax:
            return obj, ':obj:`{}.{{}}`'.format(mod)
        else:
            from docs import implemented_jax_funcs
            return implemented_jax_funcs, ':obj:`{}.{{}}`'.format(mod)


def _generate_comparison_rst(numpy_mod, brainpy_mod, jax_mod, klass=None, header=', , ', is_jax=False):
    np_obj, np_fmt = _import(numpy_mod, klass)
    np_funcs = _get_functions(np_obj)

    bm_obj, bm_fmt = _import(brainpy_mod, klass)
    bm_funcs = _get_functions(bm_obj)

    jax_obj, jax_fmt = _import(jax_mod, klass, is_jax=is_jax)
    jax_funcs = _get_functions(jax_obj)

    buf = []
    buf += [
        '.. csv-table::',
        '   :header: {}'.format(header),
        '',
    ]
    for f in sorted(np_funcs):
        np_cell = np_fmt.format(f)
        bm_cell = bm_fmt.format(f) if f in bm_funcs else r'\-'
        jax_cell = jax_fmt.format(f) if f in jax_funcs else r'\-'
        line = '   {}, {}, {}'.format(np_cell, bm_cell, jax_cell)
        buf.append(line)

    unique_names = bm_funcs - np_funcs
    for f in sorted(unique_names):
        np_cell = r'\-'
        bm_cell = bm_fmt.format(f) if f in bm_funcs else r'\-'
        jax_cell = jax_fmt.format(f) if f in jax_funcs else r'\-'
        line = '   {}, {}, {}'.format(np_cell, bm_cell, jax_cell)
        buf.append(line)

    buf += [
        '',
        '**Summary**\n',
        '- Number of NumPy functions: {}\n'.format(len(np_funcs)),
        '- Number of functions covered by ``brainpy.math``: {}\n'.format(len(bm_funcs & np_funcs)),
        '- Number of functions unique in ``brainpy.math``: {}\n'.format(len(bm_funcs - np_funcs)),
        '- Number of functions covered by ``jax.numpy``: {}\n'.format(len(jax_funcs & np_funcs)),
    ]
    return buf


def _section(header, numpy_mod, brainpy_mod, jax_mod, klass=None, is_jax=False):
    buf = [header, '-' * len(header), '', ]
    header2 = 'NumPy, brainpy.math, jax.numpy'
    buf += _generate_comparison_rst(numpy_mod, brainpy_mod, jax_mod, klass=klass, header=header2, is_jax=is_jax)
    buf += ['']
    return buf


def generate_analysis_docs():
    _write_subsections(
        module_name='brainpy.analysis',
        filename='apis/auto/analysis.rst',
        subsections={
            'Low-dimensional Analyzers': ['PhasePlane1D',
                                          'PhasePlane2D',
                                          'Bifurcation1D',
                                          'Bifurcation2D',
                                          'FastSlow1D',
                                          'FastSlow2D'],
            'High-dimensional Analyzers': ['SlowPointFinder']
        }
    )


def generate_synapses_docs():
    _write_module(module_name='brainpy.synapses',
                  filename='apis/auto/synapses.rst',
                  header='``brainpy.synapses`` module')

    _write_module(module_name='brainpy.synouts',
                  filename='apis/auto/synouts.rst',
                  header='``brainpy.synouts`` module')

    _write_module(module_name='brainpy.synplast',
                  filename='apis/auto/synplast.rst',
                  header='``brainpy.synplast`` module')


def generate_brainpy_docs():
    _write_subsections(
        module_name='brainpy',
        filename='apis/auto/brainpy.rst',
        subsections={
            'Numerical Differential Integration': ['Integrator',
                                                   'JointEq',
                                                   'IntegratorRunner',
                                                   'odeint',
                                                   'sdeint',
                                                   'fdeint'],
            'Building Dynamical System': ['DynamicalSystem',
                                          'DynSysGroup',
                                          'Sequential',
                                          'Network',
                                          'Dynamic',
                                          'Projection',
                                          ],
            'Simulating Dynamical System': ['DSRunner'],
            'Training Dynamical System': ['DSTrainer',
                                          'BPTT',
                                          'BPFF',
                                          'OnlineTrainer',
                                          'ForceTrainer',
                                          'OfflineTrainer',
                                          'RidgeTrainer'],
            'Dynamical System Helpers': ['LoopOverTime'],
        }
    )


def generate_integrators_doc():
    _write_subsections_v3(
        'brainpy._src.integrators',
        'brainpy.integrators',
        'apis/auto/integrators.rst',
        subsections={
            'ode': {'header': 'ODE integrators',
                    'content': {'base': 'Base ODE Integrator',
                                'generic': 'Generic ODE Functions',
                                'explicit_rk': 'Explicit Runge-Kutta ODE Integrators',
                                'adaptive_rk': 'Adaptive Runge-Kutta ODE Integrators',
                                'exponential': 'Exponential ODE Integrators', }},
            'sde': {'header': 'SDE integrators',
                    'content': {'base': 'Base SDE Integrator',
                                'generic': 'Generic SDE Functions',
                                'normal': 'Normal SDE Integrators',
                                'srk_scalar': 'SRK methods for scalar Wiener process'}},
            'fde': {'header': 'FDE integrators',
                    'content': {'base': 'Base FDE Integrator',
                                'generic': 'Generic FDE Functions',
                                'Caputo': 'Methods for Caputo Fractional Derivative',
                                'GL': 'Methods for Riemann-Liouville Fractional Derivative'}}

        }
    )


def generate_math_docs():
    _write_subsections_v4(
        'brainpy.math',
        'apis/math.rst',
        subsections={
            'object_base': ('Objects and Variables', 'brainpy.math'),
            'object_transform': ('Object-oriented Transformations', 'brainpy.math'),
            'environment': ('Environment Settings', 'brainpy.math'),
            # 'compat_numpy': ('Dense Operators with NumPy Syntax', 'brainpy.math'),
            # 'compat_pytorch': ('Dense Operators with PyTorch Syntax', 'brainpy.math'),
            # 'compat_tensorflow': ('Dense Operators with TensorFlow Syntax', 'brainpy.math'),
            'interoperability': ('Array Interoperability', 'brainpy.math'),
            'pre_syn_post': ('Operators for Pre-Syn-Post Conversion', 'brainpy.math'),
            'activations': ('Activation Functions', 'brainpy.math'),
            'delayvars': ('Delay Variables', 'brainpy.math'),
            'modes': ('Computing Modes', 'brainpy.math'),
            'sparse': ('``brainpy.math.sparse`` module: Sparse Operators', 'brainpy.math.sparse'),
            'event': ('``brainpy.math.event`` module: Event-driven Operators', 'brainpy.math.event'),
            'jitconn': ('``brainpy.math.jitconn`` module: Just-In-Time Connectivity Operators', 'brainpy.math.jitconn'),
            'surrogate': ('``brainpy.math.surrogate`` module: Surrogate Gradient Functions', 'brainpy.math.surrogate'),
            'random': ('``brainpy.math.random`` module: Random Number Generations', 'brainpy.math.random'),
            'linalg': ('``brainpy.math.linalg`` module: Linear algebra', 'brainpy.math.linalg'),
            'fft': ('``brainpy.math.fft`` module: Discrete Fourier Transform', 'brainpy.math.fft'),
        }
    )


def generate_algorithm_docs(path='apis/auto/algorithms/'):
    os.makedirs(path, exist_ok=True)

    module_and_name = [
        ('offline', 'Offline Training Algorithms'),
        ('online', 'Online Training Algorithms'),
        ('utils', 'Training Algorithm Utilities'),
    ]
    _write_submodules(module_name='brainpy.algorithms',
                      filename=os.path.join(path, 'algorithms.rst'),
                      header='``brainpy.algorithms`` module',
                      submodule_names=[k[0] for k in module_and_name],
                      section_names=[k[1] for k in module_and_name])


def main():
    os.makedirs('apis/auto/', exist_ok=True)

    # _write_module(module_name='brainstate.surrogate',
    #               filename='apis/auto/surrogate.rst',
    #               header='``brainstate.surrogate`` module')

    # _write_module(module_name='brainstate.random',
    #               filename='apis/auto/random.rst',
    #               header='``brainstate.random`` module')

    # _write_module(module_name='brainstate.mixin',
    #               filename='apis/auto/mixin.rst',
    #               header='``brainstate.mixin`` module')

    # _write_module(module_name='brainstate.transform',
    #               filename='apis/auto/transform.rst',
    #               header='``brainstate.transform`` module')

    # _write_module(module_name='brainstate.math',
    #               filename='apis/auto/math.rst',
    #               header='``brainstate.math`` module')

    # _write_module(module_name='brainstate.util',
    #               filename='apis/auto/util.rst',
    #               header='``brainstate.util`` module')

    # _write_module(module_name='brainstate.typing',
    #               filename='apis/typing.rst',
    #               header='``brainstate.typing`` module')

    # _write_module(module_name='brainstate.optim',
    #               filename='apis/optim.rst',
    #               header='``brainstate.optim`` module')

    # _write_module(module_name='brainstate.event',
    #               filename='apis/event.rst',
    #               header='``brainstate.event`` module')

    module_and_name = [
        ('_dict', 'Dict Operation'),
        ('_filter', 'Filter Operation'),
        ('_pretty_repr', 'Pretty Representation'),
        ('_struct', 'Struct Operation'),
        ('_visualization', 'Visualization Operation'),
        ('_others', 'Other Operations'),
    ]
    _write_submodules(module_name='brainstate.util',
                      filename='apis/util.rst',
                      header='``brainstate.util`` module',
                      submodule_names=[k[0] for k in module_and_name],
                      section_names=[k[1] for k in module_and_name])

    # module_and_name = [
    #   ('_graph_node', 'Graph Node'),
    #   ('_graph_operation', 'Graph Operation'),
    #   ('_graph_convert', 'Graph and Tree Conversion'),
    #   ('_graph_context', 'Graph Processing Context Management'),
    # ]
    # _write_submodules(module_name='brainstate.graph',
    #                   filename='apis/graph.rst',
    #                   header='``brainstate.graph`` module',
    #                   submodule_names=[k[0] for k in module_and_name],
    #                   section_names=[k[1] for k in module_and_name])

    # module_and_name = [
    #   ('_state', 'State System'),
    # ]
    # _write_submodules(module_name='brainstate',
    #                   filename='apis/brainstate.rst',
    #                   header='``brainstate`` module',
    #                   submodule_names=[k[0] for k in module_and_name],
    #                   section_names=[k[1] for k in module_and_name])

    # module_and_name = [
    #   ('_activations', 'Activation Functions'),
    #   ('_normalization', 'Normalization Functions'),
    #   ('_spikes', 'Spiking Operations'),
    # ]
    # _write_submodules(module_name='brainstate.functional',
    #                   filename='apis/functional.rst',
    #                   header='``brainstate.functional`` module',
    #                   submodule_names=[k[0] for k in module_and_name],
    #                   section_names=[k[1] for k in module_and_name])

    # module_and_name = [
    #   ('_module', 'Base Module Classes'),
    #   ('_interaction', 'Synaptic Interaction Layers'),
    #   ('_elementwise', 'Element-wise Layers'),
    #   ('_dynamics', 'Base Dynamics Classes'),
    #   ('_dyn_impl', 'Neuronal/Synaptic Dynamics'),
    #   ('_exp_euler', 'Numerical Integration Methods'),
    #   ('_collective_ops', 'Collective Operations'),
    # ]
    # _write_submodules(module_name='brainstate.nn',
    #                   filename='apis/nn.rst',
    #                   header='``brainstate.nn`` module',
    #                   submodule_names=[k[0] for k in module_and_name],
    #                   section_names=[k[1] for k in module_and_name])


if __name__ == '__main__':
    main()
