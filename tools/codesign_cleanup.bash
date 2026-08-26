#!/usr/bin/env bash
# !!! コードサイニング証明書を取り扱うので取り扱い注意 !!!

# 一時キーチェーンを破棄し、署名用Identityを削除する

set -eu

if [ ! -v P12_PATH ]; then
    echo "P12_PATHが未定義です"
    exit 1
fi
if [ ! -v CODESIGN_IDENTITY_PATH ]; then
    echo "CODESIGN_IDENTITY_PATHが未定義です"
    exit 1
fi
if [ ! -v KEYCHAIN_PATH_PATH ]; then
    echo "KEYCHAIN_PATH_PATHが未定義です"
    exit 1
fi

rm -f "$P12_PATH"

if [ -f "$KEYCHAIN_PATH_PATH" ]; then
    KEYCHAIN_PATH="$(head -n 1 "$KEYCHAIN_PATH_PATH")"

    # キーチェーンを削除
    security delete-keychain "$KEYCHAIN_PATH"
fi

# 出力ファイルを削除
rm -f "$CODESIGN_IDENTITY_PATH"
rm -f "$KEYCHAIN_PATH_PATH"
