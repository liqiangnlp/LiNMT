#!/usr/bin/perl

use strict;
use FileHandle;

if (scalar(@ARGV) eq 0 or scalar(@ARGV) % 2 ne 0)
{
  print STDERR "perl NiuTrans.NMT-observation.pl 1-Label 1-input 2-Label 2-input ...\n";
  exit(1);
}

my @labels;
my @files;

my $i = 0;
foreach my $cont (@ARGV)
{
  ++$i;
  if ($i % 2 eq 0)
  {
    push @files, $cont;
  }  
  else
  {
    push @labels, $cont;
  }

}


$i = 0;
my %fh;
foreach my $label (@labels)
{
  open ($fh{$label}, "<", $files[$i]) or die "Error: can not open $files[$i].\n";
  ++$i;
}


my $handle = $fh{$labels[0]};
my $line_num = 1;
while (my $line = <$handle>)
{
  print STDOUT "[".$line_num."]\n";
  $line =~ s/[\r\n]//g;
  $line =~ s/\s+$//g;
  $line =~ s/^\s+//g;
  my @fields = split / \|\|\|\| /, $line;
  my $value = sprintf "%-14s", $labels[0];
  print STDOUT $value.": ".$fields[0]."\n";

  foreach my $i (1..scalar(@labels) - 1)
  {
    $handle = $fh{$labels[$i]};
    $line = <$handle>;
    $line =~ s/[\r\n]//g;
    $line =~ s/\s+$//g;
    $line =~ s/^\s+//g;
    my @fields = split / \|\|\|\| /, $line;
    my $value = sprintf "%-14s", $labels[$i];
    print STDOUT $value.": ".$fields[0]."\n";
  }
  print STDOUT "\n\n";

  $handle = $fh{$labels[0]};
  ++$line_num;
}


foreach my $i (0..scalar(@labels) - 1)
{
  $handle = $fh{$labels[$i]};
  close($handle);
}







