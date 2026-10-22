"""Decode PC histograms and classify RV64IMC instructions without aliases.

Each instruction belongs to exactly one category. Loads/stores include their
compressed forms; mul and div/rem are separate; conditional branches exclude
jumps. ALU is an explicit integer mnemonic allowlist, never a catch-all. CSR,
privileged and fence instructions are system. Custom encodings use their major
opcode, regardless of the disassembler's mnemonic. Unrecognized instructions
are reported as other. Width comes from the decoded encoding, not PC spacing.
"""
import re

CATEGORIES = ('load', 'store', 'mul', 'div', 'branch', 'jump', 'alu', 'csr/system', 'custom', 'other')
GROUPS = {
    'load': 'lb lbu lh lhu lw lwu ld c.lw c.ld c.lwsp c.ldsp',
    'store': 'sb sh sw sd c.sw c.sd c.swsp c.sdsp',
    'mul': 'mul mulh mulhsu mulhu mulw',
    'div': 'div divu divw divuw rem remu remw remuw',
    'branch': 'beq bne blt bge bltu bgeu c.beqz c.bnez',
    'jump': 'jal jalr c.j c.jal c.jr c.jalr',
    'alu': ('add addw addi addiw sub subw and andi or ori xor xori sll slli sllw slliw '
            'srl srli srlw srliw sra srai sraw sraiw slt slti sltu sltiu lui auipc '
            'c.add c.addw c.addi c.addiw c.addi4spn c.addi16sp c.sub c.subw c.and '
            'c.andi c.or c.xor c.slli c.srli c.srai c.li c.lui c.mv c.nop'),
    'csr/system': 'csrrw csrrs csrrc csrrwi csrrsi csrrci ecall ebreak c.ebreak fence fence.i fence.tso wfi mret sret uret',
}
LOOKUP = {mnemonic: category for category, names in GROUPS.items() for mnemonic in names.split()}


def classify(raw, mnemonic):
    if len(raw) == 8 and int(raw, 16) & 0x7f in (0x0b, 0x2b, 0x5b, 0x7b):
        return 'custom'
    return LOOKUP.get(mnemonic, 'other')


def disassembly(text):
    instructions = {}
    for line in text.splitlines():
        match = re.match(r'^\s*([0-9a-f]+):\s+([0-9a-f]{4}|[0-9a-f]{8})\s+(\S+)', line)
        if match:
            pc, raw, mnemonic = match.groups()
            instructions[int(pc, 16)] = (raw, mnemonic, classify(raw, mnemonic))
    return instructions


def histogram(text):
    match = re.search(r'^PC Histogram size:(\d+)\n', text, re.M)
    if not match:
        raise ValueError('Missing PC histogram')
    rows = re.findall(r'^([0-9a-f]+) (\d+)$', text[match.end():], re.M)
    result = {int(pc, 16): int(count) for pc, count in rows}
    if len(rows) != len(result) or len(result) != int(match[1]):
        raise ValueError('Incomplete or duplicate PC histogram')
    return result


def self_test():
    sample = '''
    80000000: 00053503 ld a0,0(a0)
    80000004: c188 c.sw a0,0(a1)
    80000006: 02b50533 mul a0,a0,a1
    8000000a: 02b54533 div a0,a0,a1
    8000000e: c101 c.beqz a0,8000000e
    80000010: 8082 c.jr ra
    80000012: 0505 c.addi a0,1
    80000014: b0202573 csrrs a0,minstret,zero
    80000018: 0000000b .4byte 0xb
    8000001c: 00000053 fadd.s ft0,ft0,ft0
    '''
    decoded = disassembly(sample)
    if [v[2] for v in decoded.values()] != list(CATEGORIES):
        raise ValueError('Classifier sample mismatch')
    if sum(len(v[0]) // 2 for v in decoded.values()) != 32:
        raise ValueError('Instruction width mismatch')
    for category, names in GROUPS.items():
        for name in names.split():
            if classify('00000013', name) != category:
                raise ValueError(name)
    for opcode in (0x0b, 0x2b, 0x5b, 0x7b):
        if classify(f'{opcode:08x}', '.4byte') != 'custom':
            raise ValueError('Custom opcode mismatch')
    if histogram('PC Histogram size:2\n1000 1\n1004 2\n') != {4096: 1, 4100: 2}:
        raise ValueError('Histogram parser mismatch')
    print('PASS: instruction classifier, widths, custom opcodes, histogram parser')


if __name__ == '__main__':
    self_test()
