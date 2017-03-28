#!/usr/bin/perl

# * NiuTrans.NMT v1.0                   * #
# * Author: Qiang Li,                   * #
# * Email : liqiangneu@gmail.com        * #
# * Date  : 10/30/2016                  * #
# * Time  : 15:31                       * #
# * training recurrent neural network   * #


if (scalar(@ARGV) ne 2) {
  print STDERR "perl NiuTrans.NMT-pure-1best.pl INPUT OUTPUT\n";
  exit(1);
}

open (INPUT, "<", $ARGV[0]) or die "Error: can not open $ARGV[0]!\n";
open (OUTPUT, ">", $ARGV[1]) or die "Error: can not write $ARGV[1]!\n";

my $line_no = 0;
while (my $sent = <INPUT>)
{
  ++$line_no;

  $sent =~ s/[\r\n]//g;
  $sent =~ s/\s+$//g;
  $sent =~ s/^\s+//g;
  
  my @domains = split / \|\|\|\| /, $sent;
  
  print OUTPUT $domains[0]."\n";
  

  if ($line_no % 100 == 0) 
  {
    print STDERR "\r$line_no";
  }
}
print STDERR "\r$line_no\n";

close (INPUT);
close (OUTPUT);


