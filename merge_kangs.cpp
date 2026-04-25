// merge_kangs.cpp -- Merge two .kangs DP database files into one
// Part of RCKangaroo Hybrid+SOTA+
//
// Usage:  ./merge_kangs <file1.kangs> <file2.kangs> <output.kangs>
//
// Uses streaming merge: file2 is read record-by-record and inserted into
// the in-memory db1, so only ONE full TFastBase is held in RAM at a time.
// This keeps peak RAM under ~2 GB even for large kangs files.

#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include "utils.h"

static double elapsed_sec(clock_t start) {
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}

int main(int argc, char* argv[])
{
    printf("============================================================\n");
    printf("  merge_kangs -- RCKangaroo .kangs file merger\n");
    printf("============================================================\n\n");

    if (argc < 4) {
        printf("Usage: %s <file1.kangs> <file2.kangs> <output.kangs>\n\n", argv[0]);
        printf("  Merges DPs from both files. Duplicates are skipped.\n");
        printf("  file1 header is kept in the output.\n\n");
        printf("  Example:\n");
        printf("    %s A.kangs B.kangs combined.kangs\n", argv[0]);
        return 1;
    }

    const char* fn1    = argv[1];
    const char* fn2    = argv[2];
    const char* fn_out = argv[3];

    // TFastBase.lists[256][256][256] ~192 MB -- must heap-allocate
    TFastBase* db1 = new TFastBase();
    clock_t t0;

    // ---- Load file 1 fully ----
    printf("[1/4] Loading %s ...\n", fn1);
    t0 = clock();
    if (db1->LoadFromFile((char*)fn1) == false) {
        printf("ERROR: Failed to load '%s'\n", fn1);
        delete db1; return 1;
    }
    u64 cnt1 = db1->GetBlockCnt();
    printf("      %llu DPs loaded  (%.2f s)\n\n", (unsigned long long)cnt1, elapsed_sec(t0));

    // ---- Stream-merge file 2 (no second TFastBase needed) ----
    printf("[2/4] Checking %s ...\n", fn2);
    {
        FILE* fp = fopen(fn2, "rb");
        if (fp == NULL) {
            printf("ERROR: Cannot open '%s'\n", fn2);
            delete db1; return 1;
        }
        fclose(fp);
    }

    printf("[3/4] Streaming merge from %s ...\n", fn2);
    t0 = clock();
    u64 added = db1->MergeFromFile(fn2);
    u64 total = db1->GetBlockCnt();
    u64 dupes = (total - cnt1 == added) ? (u64)0 : (total - cnt1 - added); // safety
    (void)dupes;
    printf("      Added  : %llu new DPs\n",  (unsigned long long)added);
    printf("      Total  : %llu DPs  (%.2f s)\n\n", (unsigned long long)total, elapsed_sec(t0));

    // ---- Save ----
    printf("[4/4] Saving to %s ...\n", fn_out);
    t0 = clock();
    if (db1->SaveToFile((char*)fn_out) == false) {
        printf("ERROR: Failed to save '%s'\n", fn_out);
        delete db1; return 1;
    }
    printf("      Saved %llu DPs  (%.2f s)\n\n", (unsigned long long)total, elapsed_sec(t0));

    delete db1;

    printf("============================================================\n");
    printf("  Done. %llu DPs in %s\n", (unsigned long long)total, fn_out);
    printf("============================================================\n");
    return 0;
}
