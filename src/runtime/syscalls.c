#include "htif.h"
#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/stat.h>
#include <sys/types.h>

extern char __heap_start[], __heap_end[];

void *_sbrk(ptrdiff_t increment) {
    static uintptr_t current;
    const uintptr_t start = (uintptr_t)__heap_start;
    const uintptr_t end = (uintptr_t)__heap_end;
    if (current == 0) current = start;
    uintptr_t previous = current;
    if (increment >= 0) {
        if ((uintptr_t)increment > end - current) goto exhausted;
        current += (uintptr_t)increment;
    } else {
        uintptr_t amount = (uintptr_t)(-(increment + 1)) + 1;
        if (amount > current - start) goto exhausted;
        current -= amount;
    }
    return (void *)previous;
exhausted:
    errno = ENOMEM;
    return (void *)-1;
}

int _write(int fd, const void *buffer, size_t length) {
    if (fd != 1 && fd != 2) { errno = EBADF; return -1; }
    const unsigned char *bytes = buffer;
    for (size_t i = 0; i < length; ++i) htif_putchar(bytes[i]);
    return (int)length;
}
_Noreturn void _exit(int code) { htif_exit(code); }
int _close(int fd) { (void)fd; errno = EBADF; return -1; }
int _fstat(int fd, struct stat *st) {
    if (fd < 0 || fd > 2) { errno = EBADF; return -1; }
    st->st_mode = S_IFCHR;
    return 0;
}
int _isatty(int fd) {
    if (fd >= 0 && fd <= 2) return 1;
    errno = EBADF;
    return 0;
}
off_t _lseek(int fd, off_t offset, int whence) {
    (void)fd; (void)offset; (void)whence;
    errno = ESPIPE;
    return (off_t)-1;
}
int _read(int fd, void *buffer, size_t length) {
    (void)buffer; (void)length;
    if (fd == 0) return 0;
    errno = EBADF;
    return -1;
}
int _kill(int pid, int signal) {
    (void)pid; (void)signal; errno = EINVAL; return -1;
}
int _getpid(void) { return 1; }
