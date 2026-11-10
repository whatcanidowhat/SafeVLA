"""Apply a single declared filesystem-monitoring fix to the frozen pair runner.

Original files stay intact. Commands, ON/OFF ordering, validators, deadlines and
90-GiB/10-GiB disk limits are executed from the original module unchanged.
"""
import ast
from pathlib import Path

from output_scan import used_directory_bytes


def load_patched_pair():
    from diagnostics.actor_memory_unmask import online_pair

    source = Path(online_pair.__file__).read_text()
    tree = ast.parse(source, filename=online_pair.__file__)
    old_body = ast.parse(
        "def used_bytes():\n"
        "    return sum(x.stat().st_size for x in a.output.rglob('*') if x.is_file())\n"
    ).body[0].body
    new_body = ast.parse(
        "def used_bytes():\n"
        "    return _unmask_safe_directory_bytes(a.output)\n"
    ).body[0].body

    class ReplaceOutputScan(ast.NodeTransformer):
        count = 0

        def visit_FunctionDef(self, node):
            if node.name != 'used_bytes':
                return self.generic_visit(node)
            if ([ast.dump(x) for x in node.body] != [ast.dump(x) for x in old_body]
                    or node.args.args or node.decorator_list):
                raise RuntimeError('Unexpected original output-size guard; refusing patch')
            self.count += 1
            replacement = ast.increment_lineno(new_body[0], node.body[0].lineno - new_body[0].lineno)
            node.body = [ast.copy_location(replacement, node.body[0])]
            return node

    patch = ReplaceOutputScan()
    tree = patch.visit(tree)
    if patch.count != 1:
        raise RuntimeError('Expected exactly one frozen used_bytes() function')
    ast.fix_missing_locations(tree)
    online_pair.__dict__['_unmask_safe_directory_bytes'] = used_directory_bytes
    # __name__ remains the original module name, so its main guard is not run.
    exec(compile(tree, online_pair.__file__, 'exec'), online_pair.__dict__)
    return online_pair


def main():
    print('Output guard: tolerate disappearing descendants during atomic episode completion.', flush=True)
    load_patched_pair().main()


if __name__ == '__main__':
    main()
